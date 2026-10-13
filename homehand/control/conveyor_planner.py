"""Conveyor controller: perceive at 5 Hz, choose the next object and hand, intercept it on the belt, drop it in the bin.

Both hands work at once, each in its own pick zone and dropping into the bin at its own hip; the arm command merges
the arm joints and hand of each busy hand's skill. Choosing, for an idle hand: for every tracked object not yet
taken, the earliest synchronisation
time t_sync such that
- the arm can reach the hover pose by then (joint-space path, peak joint speed within the limit),
- the object is inside that hand's pick zone at t_sync and still inside when the synced phases end
  (`conveyor_skill.SYNC_TOTAL` control steps later) - the belt runs towards +y, through the right hand's zone first,
- the grasp pose (side grasp upright, from above lying) has an IK solution within 1 cm at t_sync.
The candidate with the earliest t_sync wins. Objects that no hand can take are left to ride on (logged
`not_reachable`); a grasp that closes on nothing releases the object for another try.
"""

from __future__ import annotations

import numpy as np

from homehand.control.conveyor_skill import HOVER_H, SYNC_TOTAL, ConveyorPickSkill, target_from_pose
from homehand.control.grasps import RELAXED_HAND
from homehand.control.ik import ArmIK
from homehand.control.safety import ARM_MAX_VEL
from homehand.control.skills import compose_action, grasp_plan, grasp_variants
from homehand.model import spec
from homehand.model.build import CONVEYOR_BINS
from homehand.perception.conveyor_perception import ConveyorPerception

PERCEIVE_EVERY = 5                      # control steps (5 Hz)
ZONE = (0.10, 0.36)                     # |y| of each hand's pick zone (the G1 reaches |y| <= 0.38, not the chest)
# each hand drops into the bin at its own hip, on its inner half (the palm-down hand releases up to |y| ~0.35); successive drops
# take these spots in turn: (x offset, offset towards the body's midline) from the bin's centre
DROP_OFFSETS = ((0.0, 0.06), (0.05, 0.05), (-0.05, 0.05), (0.0, 0.04))
MIN_REACH_STEPS = 12
IK_TOL = 0.01
SCAN = 0.25                             # s between candidate synchronisation times
RECHECK = 1.0                           # s before an infeasible object / hand pair is evaluated again
LOOKAHEAD = 4.0                         # s ahead in which synchronisation times are considered


class ConveyorPlanner:
    kind = "conveyor"

    def __init__(self, env, perception: str = "oracle", ik: ArmIK | None = None, sides=spec.SIDES):
        """sides: the hands that pick (both by default)."""
        self.env = env
        self.sides = tuple(sides)
        self.ik = ik or ArmIK()
        self.perception = ConveyorPerception(env, perception)
        self.perception_mode = perception
        self.skill_kind = "conveyor_expert"
        self.last_perception = None

    def reset(self, obs: dict) -> None:
        self.speed = float(obs["belt"]["speed"])
        self.perception.reset((0.0, self.speed))
        self.q = self.ik.q_home.copy()
        self.hand_cmd = {s: np.array(RELAXED_HAND, float) for s in spec.SIDES}
        self.skills: dict[str, ConveyorPickSkill | None] = {s: None for s in spec.SIDES}
        self.taken: set[str] = set()
        self.infeasible_until: dict = {}
        self.skipped: set[str] = set()
        self.drops = {s: 0 for s in spec.SIDES}
        self.log: list[dict] = []
        self.rejects: dict = {}                    # (object, side, reason) -> count, why candidates failed
        self.order: list[str] = []                 # objects in the order they came into view
        self.left_ok: dict[str, bool] = {}         # can the left hand take it in its own zone (checked once)
        self.done_flag = False

    @property
    def done(self) -> bool:
        return self.done_flag

    @property
    def phase(self) -> str:
        busy = [f"{s}:{sk.phase}" for s, sk in self.skills.items() if sk is not None]
        return busy[0] if busy else "wait"

    @property
    def skill(self):                     # (the first busy hand's skill: for logging and the viewer)
        return next((sk for sk in self.skills.values() if sk is not None), None)

    # ------------------------------------------------------------------ decision
    def _zone(self, side: str) -> tuple[float, float]:
        return (-ZONE[1], -ZONE[0]) if side == "right" else ZONE

    def _reach_steps(self, side: str, hover: np.ndarray, quat) -> int:
        path = self.ik.plan_joint_path(self.q, side, [(hover, quat)])
        dq = float(np.abs(path[-1] - path[0]).max())
        return max(MIN_REACH_STEPS, int(np.ceil(1.875 * dq / (0.8 * ARM_MAX_VEL * spec.CONTROL_DT))) + 2)

    def _candidate(self, name: str, side: str, now: float):
        """(earliest feasible t_sync, grasp variant) for this object and hand (t_sync scanned every SCAN s through
        the pick zone; at each, the default grasp first, then the alternative), or None."""
        track = self.perception.tracks[name]
        lo, hi = self._zone(side)
        v = float(track.velocity[1])
        y_now = float(track.predict(now).centre[1])
        sync_dur = SYNC_TOTAL * spec.CONTROL_DT
        if v <= 1e-4:                                         # belt stopped: now or never
            if not lo <= y_now <= hi:
                return None
            times = [now + MIN_REACH_STEPS * spec.CONTROL_DT]
        else:
            t0 = max(now + MIN_REACH_STEPS * spec.CONTROL_DT, now + (lo - y_now) / v)
            t1 = now + (hi - y_now) / v - sync_dur               # the synced phases must end inside the zone
            # only the next LOOKAHEAD s: further out there is time to decide later (scanning a far object's whole
            # passage was hundreds of IK solves per look)
            times = list(np.arange(t0, min(t1, now + LOOKAHEAD) + 1e-9, SCAN))
        for t_sync in times:
            pose = track.predict(t_sync)
            for variant in grasp_variants(target_from_pose(pose, self.env.obj_info, self.perception.support_z)):
                if self._feasible(name, side, now, t_sync, pose, variant):
                    return float(t_sync), variant
        return None

    def _feasible(self, name: str, side: str, now: float, t_sync: float, pose, variant: int) -> bool:
        tg = target_from_pose(pose, self.env.obj_info, self.perception.support_z, variant)
        p = grasp_plan(side, tg)
        # the grasp, the pre-grasp and the hover pose must all be reachable (behind an object close to the
        # chest the pre-grasp of a side grasp is not: the reach ended 5 cm off and the IK fought the arm)
        probe = ConveyorPickSkill.__new__(ConveyorPickSkill)
        probe.top, probe.target, probe.support_z = p["approach"] == "top", tg, self.perception.support_z
        pre, hover = probe._pre(p), probe._hover(p)
        bad = None
        for what, pos in (("grasp", p["grasp"]), ("pre", pre), ("hover", hover)):
            q, err = self.ik.solve(self.ik.q_work, {side: (pos, p["quat"])}, iters=120)
            if err[side] > IK_TOL:
                bad = what
                break
        if bad is None and now + self._reach_steps(side, hover, p["quat"]) * spec.CONTROL_DT > t_sync:
            bad = "time"                                  # the arm cannot be there in time
        if bad is not None:
            key = (name, side, bad)
            self.rejects[key] = self.rejects.get(key, 0) + 1
            return False
        return True

    def _left_can_take(self, name: str, now: float) -> bool:
        """Whether the left hand reaches the object (grasp, pre-grasp, hover) when it is in the middle of the left
        hand's zone, at its current heading."""
        if name not in self.left_ok:
            track = self.perception.tracks[name]
            pose = track.predict(now)
            shift = np.array([0.0, float(np.mean(ZONE)) - float(pose.centre[1]), 0.0])
            pose = type(pose)(pose.name, pose.pos + shift, pose.R, pose.rmse, pose.inliers, pose.n_points, pose.rest,
                              pose.heading, pose.centre + shift)
            ok = False
            for variant in grasp_variants(target_from_pose(pose, self.env.obj_info, self.perception.support_z)):
                tg = target_from_pose(pose, self.env.obj_info, self.perception.support_z, variant)
                p = grasp_plan("left", tg)
                probe = ConveyorPickSkill.__new__(ConveyorPickSkill)
                probe.top, probe.target, probe.support_z = p["approach"] == "top", tg, self.perception.support_z
                ok = ok or all(
                    self.ik.solve(self.ik.q_work, {"left": (pos, p["quat"])}, iters=120)[1]["left"] <= IK_TOL
                    for pos in (p["grasp"], probe._pre(p), probe._hover(p)))
            self.left_ok[name] = ok
        return self.left_ok[name]

    def _choose(self, now: float, sides):
        best = None
        for name, track in self.perception.tracks.items():
            if name not in self.order:
                self.order.append(name)
            if name in self.taken or track.n_updates < 2:
                continue
            for side in sides:
                # the hands take turns: every second object rides on to the left hand (the belt brings everything
                # past the right hand first, which would otherwise take nearly all), unless the left cannot take it
                if (side == "right" and len(self.sides) == 2 and self.order.index(name) % 2 == 1
                        and self._left_can_take(name, now)):
                    continue
                key = (name, side)
                # an object/hand pair found infeasible is looked at again only RECHECK s later (each look is up
                # to a few dozen IK solves)
                if now < self.infeasible_until.get(key, -1.0):
                    continue
                cand = self._candidate(name, side, now)
                if cand is None:
                    self.infeasible_until[key] = now + RECHECK
                elif best is None or cand[0] < best[0]:
                    best = (cand[0], name, side, cand[1])
        return best

    def _others(self, name: str, t: float) -> list:
        """The other tracked objects (predicted at time t) as obstacles."""
        return [target_from_pose(tr.predict(t), self.env.obj_info, self.perception.support_z)
                for n, tr in self.perception.tracks.items() if n != name and n not in self.taken]

    # ------------------------------------------------------------------ loop
    def _start(self, choice, obs: dict, t: float) -> None:
        t_sync, name, side, variant = choice
        track = self.perception.tracks[name]
        tg = target_from_pose(track.predict(t_sync), self.env.obj_info, self.perception.support_z, variant)
        dx, din = DROP_OFFSETS[self.drops[side] % len(DROP_OFFSETS)]
        bx, by = CONVEYOR_BINS[side]
        drop = np.array([bx + dx, by - np.sign(by) * din])
        self.drops[side] += 1
        self.skills[side] = ConveyorPickSkill(self.ik, side, name, track.predict, self.q, self.hand_cmd, t, t_sync,
                                              track.velocity, self.env.obj_info, self.perception.support_z, drop,
                                              others=lambda tt, name=name: self._others(name, tt), variant=variant)
        self.taken.add(name)
        self.log.append({"t": obs["t"], "event": "pick", "object": name, "side": side,
                         "t_sync": round(t_sync, 2), "rest": tg.rest, "grasp": variant})

    def act(self, obs: dict) -> np.ndarray:
        t = obs["t"] * spec.CONTROL_DT
        if obs["t"] % PERCEIVE_EVERY == 0:
            self.perception.update(t)
            for side in self.sides:                         # an idle hand takes the next object it can
                if self.skills[side] is None:
                    choice = self._choose(t, [side])
                    if choice is not None:
                        self._start(choice, obs, t)
        for side, sk in self.skills.items():
            if sk is None:
                continue
            sk.act(obs)
            idx = self.ik.arm_qadr[side]                    # this arm and hand come from this hand's skill
            self.q[idx] = sk.q[idx]
            self.hand_cmd[side] = np.asarray(sk.hand_cmd[side], float).copy()
            if sk.done:
                self.log.append({"t": obs["t"], "event": "skill_done", "object": sk.name, "side": side,
                                 "status": "placed" if sk.placed else sk.status})
                if sk.status in ("empty_grasp", "aborted"):      # free for another try
                    self.taken.discard(sk.name)
                self.skills[side] = None
        busy = [s for s, sk in self.skills.items() if sk is not None]
        self.env.context = {"phase": self.phase, "target": self.skills[busy[0]].name if busy else None}
        # the episode ends when the belt has delivered everything and nothing is left on it
        if not busy and not getattr(self.env, "queue", []) and not any(o["on_table"] for o in obs["objects"].values()):
            self.done_flag = True
        return compose_action(self.ik, self.q, self.hand_cmd)
