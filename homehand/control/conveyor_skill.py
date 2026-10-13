"""Picking an object off the running belt: reach where it will be, move with it while grasping, then carry it away.

1. reach   joint-space swing (as in `PickPlaceSkill`) to a hover pose above where the tracker predicts the object at
           `t_sync`, timed to arrive then.
2. synced  descend -> approach -> close -> lift, a fixed number of control steps each, in the object's moving frame:
           the palm target of every step is the grasp plan (`skills.grasp_plan`: side grasp for an upright object, from
           above for a lying one) on the object pose the tracker predicts for that step (constant belt velocity). As
           in the tidy-table expert, the whole joint path is solved before it runs (`ArmIK.plan_path`: converged,
           warm-started IK) - tracking with step-by-step differential IK lost the arm's branch and dived to the belt.
           Closed loop: at every perception update the prediction is compared with the one the path was planned on;
           more than REPLAN_TOL apart, the rest of the synced path is planned again from the current configuration.
           The lift keeps the belt's velocity (no shearing pull on the belt) while it rises.
3. carry   from the lifted pose the remaining phases of the proven tidy-table skill (`PickPlaceSkill`: transport,
           drop, straight retreat, return), planned from the current configuration.

A side grasp that closes on nothing (proprioceptive, as in `PickPlaceSkill`) opens, rises and goes home.
"""

from __future__ import annotations

import numpy as np

from homehand.control.grasps import RELAXED_HAND
from homehand.control.ik import ArmIK
from homehand.control.safety import ARM_MAX_VEL
from homehand.control.skills import (REACH_Z, TOP_SWING_Z, PickPlaceSkill, Target, compose_action, grasp_plan,
                                     min_jerk)
from homehand.model import spec
from homehand.model.objects import lying_dims

PHASES = (("descend", 12), ("approach", 12), ("close", 18), ("lift", 15))   # minimum control steps per phase
SYNC_TOTAL = 90            # control steps the planner budgets for the synced phases (they stretch to the joint limits)
HOVER_H = 0.06             # m above the pre-grasp pose where the reach ends (a short, slow descent in sync)
JOINT_FRAC = 0.8           # of the safety layer's joint speed limit, per synced step
LIFT_H = 0.18              # m: the carried object clears the others on the belt before the arm swings to its bin
REPLAN_TOL = 0.01          # m between the current prediction and the one the synced path was planned on
CHECK_EVERY = 5            # control steps (the perception rate)
MAX_REPLANS = 2
SYNC_IK_ITERS = 25
SIDE_EXTRA_BACK = 0.03     # m: side grasps start their approach this much further behind the object
HOVER_ABOVE_TOP = 0.08     # m: palm height above an upright object's top where a side grasp's descent starts
MAX_SYNC_STEPS = 2 * SYNC_TOTAL   # a synced path longer than this, or off its targets by more than SYNC_IK_TOL,
SYNC_IK_TOL = 0.02                # aborts the pick (the object rode out of reach: the stretched descent followed a can
                                  # for 24 s across the zone and the arm finally flew up out of the workspace)
DROP_RIM = 0.90            # m: release as above a rim this high (the bins' is 0.75: the objects drop into the deep bin)


def _along(path: np.ndarray, u: float) -> np.ndarray:
    """The joint configuration a fraction u along a sampled path, linearly interpolated between its samples (taking
    the nearest sample made the arm move in steps: 50-67 rad/s^2 joint acceleration peaks during the reach)."""
    x = float(np.clip(u, 0.0, 1.0)) * (len(path) - 1)
    i = min(int(x), len(path) - 2)
    return path[i] + (path[i + 1] - path[i]) * (x - i)


def target_from_pose(pose, obj_info: dict, support_z: float, variant: int = 0) -> Target:
    """The grasp library's view of a 6-D pose: footprint centre, heading, height, rest pose (+ which grasp)."""
    h = lying_dims(pose.name)["thick"] if pose.rest == "lying" else float(obj_info[pose.name]["height"])
    return Target(pose.name, np.array([pose.centre[0], pose.centre[1], support_z]), float(pose.heading), h,
                  rest=pose.rest, variant=variant if pose.rest == "lying" else 0)


class ConveyorPickSkill:
    def __init__(self, ik: ArmIK, side: str, name: str, predict, q_start: np.ndarray, hand_cmd: dict,
                 now: float, t_sync: float, belt_velocity, obj_info: dict, support_z: float, drop_xy=None,
                 others=None, variant: int = 0):
        """`predict(t)` -> the tracker's Pose6D of the object at time t (seconds); `others(t)` -> the Targets of the
        other objects on the belt at time t (obstacles for the carry)."""
        self.others = others
        self.variant = variant
        self.ik, self.side, self.name, self.predict = ik, side, name, predict
        self.obj_info, self.support_z, self.drop_xy = obj_info, support_z, drop_xy
        self.q = q_start.copy()
        self.hand_cmd = {s: np.asarray(v, float).copy() for s, v in hand_cmd.items()}
        self.status = "running"
        self.steps, self.replans = 0, 0
        self.carry = None
        self.t_sync = t_sync
        self.target = self._target(t_sync)
        p = grasp_plan(side, self.target)
        self.grasp_type = p["grasp_type"]
        self.top = p["approach"] == "top"
        self.open_ = np.asarray(self.grasp_type.open_cmd(), float)
        self.closed = np.asarray(self.grasp_type.closed_cmd(), float)
        n = max(10, int(round((t_sync - now) / spec.CONTROL_DT)))
        path = ik.plan_joint_path(self.q, side, [(self._hover(p), p["quat"])])
        self.reach = [_along(path, min_jerk((i + 1) / n)) for i in range(n)]
        self.hand_from = self.hand_cmd[side].copy()
        self.k = 0                                    # step within reach, then within the synced path
        self.sync = None                              # [(phase, k, n, q, planned object centre xy)] per synced step
        self.counts = dict(PHASES)

    # ------------------------------------------------------------------ geometry
    def _target(self, t: float) -> Target:
        tg = target_from_pose(self.predict(t), self.obj_info, self.support_z, self.variant)
        tg.drop_xy = self.drop_xy
        tg.bin_top = DROP_RIM
        return tg

    def _pre(self, p: dict) -> np.ndarray:
        if "pre" in p:                                   # from above
            return p["pre"]
        # side grasp: further back than on the counter (the open fingers reach ~11 cm in front of the palm)
        return p["grasp"] + np.array([*p["back"], 0.0]) * (1.0 + SIDE_EXTRA_BACK / np.linalg.norm(p["back"]))

    def _hover(self, p: dict) -> np.ndarray:
        pre = self._pre(p)
        z = pre[2] + HOVER_H
        if not self.top:   # the open fingers come down behind the object, never onto its top
            z = max(z, self.support_z + self.target.height + HOVER_ABOVE_TOP)
        return np.array([pre[0], pre[1], z])

    def _plan_sync(self, q0: np.ndarray, t0: float, start: int) -> None:
        """Joint path of the synced steps from `start` (step i at t0 + (i - start) * dt). Each phase gets at least
        its minimum steps and more where the arm would exceed JOINT_FRAC of its speed limit; IK is warm-started
        from the previous step only (never re-solved from another posture: a branch jump of the 7-DoF arm cannot
        be executed in the time the moving object leaves)."""
        counts = dict(PHASES) if self.sync is None else dict(self.counts)
        for _ in range(2):
            plan, err = self._solve_sync(q0, t0, start, counts)
            idx = self.ik.arm_qadr[self.side]
            qs = np.array([q0] + [s[3] for s in plan])
            dq = np.abs(np.diff(qs[:, idx], axis=0)).max(axis=1)
            limit = JOINT_FRAC * ARM_MAX_VEL * spec.CONTROL_DT
            grow = {}
            for (phase, *_), d in zip(plan, dq):
                grow[phase] = max(grow.get(phase, 0.0), d / limit)
            if all(g <= 1.0 for g in grow.values()):
                break
            for phase, g in grow.items():
                if g > 1.0:            # (capped: a larger excess is a branch jump, not a matter of time)
                    counts[phase] = int(np.ceil(counts[phase] * min(g, 3.0)))
        self.counts = counts
        self.sync = (self.sync[:start] if self.sync else []) + plan
        if len(self.sync) > MAX_SYNC_STEPS or err > SYNC_IK_TOL:
            self.status = "aborted"

    def _solve_sync(self, q0: np.ndarray, t0: float, start: int, counts: dict) -> list:
        plan, q, i, worst = [], q0.copy(), 0, 0.0
        other = self.ik.arm_qadr["right" if self.side == "left" else "left"]
        for phase, _ in PHASES:
            n = counts[phase]
            for k in range(n):
                if i >= start:
                    t = t0 + (i - start) * spec.CONTROL_DT
                    tg = self._target(t)
                    p = grasp_plan(self.side, tg)
                    pre = self._pre(p)
                    u = min_jerk((k + 1) / n)
                    hover = self._hover(p)
                    pos = {"descend": hover + (pre - hover) * u, "approach": pre + (p["grasp"] - pre) * u,
                           "close": p["grasp"], "lift": p["grasp"] + np.array([0.0, 0.0, LIFT_H * u])}[phase]
                    # consecutive targets are millimetres apart: a few warm-started iterations converge (each one is
                    # a QP with the collision constraints: 100 of them per step made one plan take minutes)
                    q, err = self.ik.solve(q, {self.side: (pos, p["quat"])}, iters=SYNC_IK_ITERS)
                    worst = max(worst, err[self.side])
                    q[other] = q0[other]
                    plan.append((phase, k, n, q.copy(), tg.pos[:2].copy()))
                i += 1
        return plan, worst

    # ------------------------------------------------------------------ execution
    @property
    def done(self) -> bool:
        return self.carry is not None and self.carry.done

    @property
    def phase(self) -> str:
        if self.carry is not None:
            return self.carry.phase
        return "reach" if self.sync is None else self.sync[min(self.k, len(self.sync) - 1)][0]

    @property
    def placed(self) -> bool:
        return self.carry is not None and self.carry.status == "placed"

    def finger_flexion(self, obs) -> float:
        base = 0 if self.side == "left" else 6
        return float(np.mean(obs["hand_q"][base + 2: base + 4]))

    def act(self, obs: dict) -> np.ndarray:
        self.steps += 1
        if self.carry is not None:
            a = self.carry.act(obs)
            self.q, self.hand_cmd = self.carry.q, self.carry.hand_cmd
            return a
        t = obs["t"] * spec.CONTROL_DT
        if self.sync is None:                                   # reaching
            self.q = self.reach[self.k].copy()
            u = (self.k + 1) / len(self.reach)
            self.hand_cmd[self.side] = self.hand_from + (self.open_ - self.hand_from) * min(1.0, 1.5 * u)
            self.k += 1
            if self.k >= len(self.reach):
                self._plan_sync(self.q, t + spec.CONTROL_DT, 0)
                self.k = 0
                if self.status == "aborted":
                    self._start_carry()
            return compose_action(self.ik, self.q, self.hand_cmd)
        phase, k, n, q, planned_c = self.sync[self.k]
        if phase in ("descend", "approach") and obs["t"] % CHECK_EVERY == 0:
            now_c = self.predict(t).centre[:2]
            if np.linalg.norm(now_c - planned_c) > REPLAN_TOL and self.replans < MAX_REPLANS:   # not where planned
                self._plan_sync(self.q, t, self.k)
                self.replans += 1
                if self.status == "aborted":
                    self._start_carry()
                    return compose_action(self.ik, self.q, self.hand_cmd)
                phase, k, n, q, planned_c = self.sync[self.k]
        self.q = q.copy()
        if phase == "close":
            self.hand_cmd[self.side] = self.open_ + (self.closed - self.open_) * min(1.0, (k + 1) / (0.7 * n))
        elif phase == "lift" and k == 0:
            if not self.top and self.finger_flexion(obs) > self.grasp_type.empty_threshold():
                self.status = "empty_grasp"
                self.hand_cmd[self.side] = self.open_.copy()
        self.k += 1
        if self.k >= len(self.sync):
            self._start_carry()
        return compose_action(self.ik, self.q, self.hand_cmd)

    def _start_carry(self) -> None:
        """The tidy-table skill's remaining phases from here: transport, drop, retreat, return (or just return
        after an empty grasp)."""
        tg = self.target = self._target(self.t_sync + len(self.sync) * spec.CONTROL_DT)
        t_end = self.t_sync + len(self.sync) * spec.CONTROL_DT
        obstacles = self.others(t_end) if self.others else None
        sk = PickPlaceSkill(self.ik, self.side, tg, self.q, self.hand_cmd, obstacles=obstacles)
        empty = self.status in ("empty_grasp", "aborted")
        keep = ("return",) if empty else ("transport", "drop", "retreat", "return")
        sk.plan = [s for s in sk.plan if s.name in keep]
        for seg in sk.plan:
            # the palm follows a Cartesian path to the bin: interpolating the joints from the lifted pose to the
            # drop pose dipped the hand 15 cm, the fingers onto the belt
            if seg.name == "transport":
                seg.joint_space = False
        if empty:
            sk.hand_cmd[self.side] = np.array(RELAXED_HAND, float)
        sk.q = self.q.copy()
        sk._start()
        self.carry = sk
