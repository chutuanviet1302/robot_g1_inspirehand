"""Table-tidying task planner: perceive -> pick an object and a hand -> run the pick&place skill -> repeat.

Hand assignment follows the task rule: an object is taken by the hand it is closest to. When both hands have
work, they alternate. After every skill the head camera looks again, so the plan adapts to whatever actually
happened (dropped objects, missed grasps, objects pushed by the other hand).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from homehand.control.grasps import GRASPS, LYING_GRASP, RELAXED_HAND
from homehand.control.ik import ArmIK
from homehand.control.skills import PickPlaceSkill, Target, compose_action, grasp_plan
from homehand.env.kitchen_env import BIN_HALF_INNER, BIN_WALL_TOP, BIN_XY, KitchenEnv
from homehand.model import spec
from homehand.model.objects import OBJECTS, lying_dims

PERCEPTION_MODES = ("ground_truth", "oracle", "yolo", "auto")
BIN_CLEARANCE = 0.008    # m: finger half-thickness around the fingertip sites


@dataclass
class PerceptionResult:
    t: int
    mode: str
    targets: list[Target]
    detections: list[dict] = field(default_factory=list)   # 2D boxes for the UI
    latency_ms: float = 0.0


class TidyPlanner:
    def __init__(self, env: KitchenEnv, perception: str = "auto", skill: str = "expert", policy=None,
                 max_attempts: int = 2, ik: ArmIK | None = None, collect: bool = False):
        self.env = env
        self.collect = collect  # record (features, action) of every skill execution for imitation learning
        self.ik = ik or ArmIK()
        self.perception_mode = perception
        self.policy = policy
        self.skill_kind = "expert" if policy is None else policy.kind
        self.max_attempts = max_attempts
        self.camera = None
        self.detector = None
        if perception != "ground_truth":
            from homehand.perception.camera import HeadCamera
            from homehand.perception.detector import make_detector
            self.camera = HeadCamera(env)
            self.detector = make_detector(perception, env)
            self.perception_mode = self.detector.kind
        self.home_xy = {s: self.ik.palm_pose(self.ik.q_home, s)[0][:2] for s in spec.SIDES}

    # ------------------------------------------------------------------ lifecycle
    def reset(self, obs: dict) -> None:
        self.q = self.ik.q_home.copy()
        self.hand_cmd = {s: np.array(RELAXED_HAND, float) for s in spec.SIDES}
        self.state = "perceive"
        self.skill = None
        self.attempts: dict[str, int] = {}
        self.last_side: str | None = None
        self.log: list[dict] = []
        self.last_perception: PerceptionResult | None = None
        self.current: dict | None = None
        self.demos: list[dict] = []      # finished skill executions (collect=True)
        self.in_bin_xy: list[np.ndarray] = []
        self.used_spots: list[np.ndarray] = []
        layout_key = [round(float(v), 3) for o in obs.get("objects", {}).values() for v in o["pos"][:2]]
        self.rng = np.random.default_rng(abs(hash(tuple(layout_key))) % 2**32)   # varies with the layout
        self._demo: dict | None = None

    @property
    def done(self) -> bool:
        return self.state == "done"

    @property
    def phase(self) -> str:
        if self.state == "execute" and self.skill is not None:
            return f"{self.current['side']}:{self.skill.phase}"
        return self.state

    def act(self, obs: dict) -> np.ndarray:
        if self.state == "perceive":
            self.last_perception = self.perceive(obs)
            choice = self.choose(self.last_perception.targets)
            if choice is None:
                self.state = "done"
                self.log.append({"t": obs["t"], "event": "table_clear" if not self.last_perception.targets
                                 else "gave_up", "remaining": [t.name for t in self.last_perception.targets]})
            else:
                side, target = choice
                self.choose_drop_spot(side, target)
                self.attempts[target.name] = self.attempts.get(target.name, 0) + 1
                self.current = {"side": side, "target": target.to_dict(), "attempt": self.attempts[target.name],
                                "grasp": GRASPS[LYING_GRASP[target.name] if target.rest == "lying"
                                                else OBJECTS[target.name].grasp].label}
                self.log.append({"t": obs["t"], "event": "pick", **self.current})
                self.skill = self._make_skill(side, target, obs)
                self.last_side = side
                self.state = "execute"
        if self.state == "execute":
            feats = None
            if self.collect:
                from homehand.policy.features import skill_features
                feats = skill_features(obs, self.current["side"], self.skill.target, self.skill.steps)
            a = self.skill.act(obs)
            if self.collect:
                from homehand.policy.features import side_action
                if self._demo is None:
                    self._demo = {"side": self.current["side"], "object": self.skill.target.name, "obs": [], "action": []}
                self._demo["obs"].append(feats)
                self._demo["action"].append(side_action(a, self.current["side"]))
            if self.skill.done:
                if self._demo is not None:
                    self._demo["status"] = self.skill.status
                    self.demos.append(self._demo)
                    self._demo = None
                self.log.append({"t": obs["t"], "event": "skill_done", "side": self.current["side"],
                                 "object": self.current["target"]["name"], "status": self.skill.status})
                self.q = self.skill.q
                self.hand_cmd = self.skill.hand_cmd
                self.skill = None
                self.state = "perceive"
            return a
        return compose_action(self.ik, self.q, self.hand_cmd)

    def _make_skill(self, side: str, target: Target, obs: dict):
        if self.skill_kind != "expert":  # learned policy (ACT / Diffusion Policy)
            from homehand.policy.skill import PolicySkill
            return PolicySkill(self.policy, self.ik, side, target, self.q, self.hand_cmd)
        return PickPlaceSkill(self.ik, side, target, self.q, self.hand_cmd)

    # ------------------------------------------------------------------ perception
    def perceive(self, obs: dict) -> PerceptionResult:
        t0 = time.perf_counter()
        if self.perception_mode == "ground_truth":
            targets = []
            for name, o in obs["objects"].items():
                if not o["on_table"]:
                    continue
                if o["rest"] == "lying":
                    c = o["centre"]
                    targets.append(Target(name, np.array([c[0], c[1], o["pos"][2]]), o["yaw"],
                                          lying_dims(name)["thick"], rest="lying"))
                else:
                    targets.append(Target(name, o["pos"].copy(), o["yaw"], float(self.env.obj_info[name]["height"])))
            dets = []
        else:
            from homehand.perception.localize import localize
            frame = self.camera.capture(with_seg=self.detector.kind == "oracle")
            dets, targets = [], []
            for det in self.detector.detect(frame):
                est = localize(det, frame)
                if est is None:
                    continue
                dets.append(est.to_dict())
                targets.append(Target(est.name, est.pos, est.yaw, est.height, rest=est.rest))
        self.in_bin_xy = [t.pos[:2].copy() for t in targets if self._in_bin(t)]
        targets = [t for t in targets if self._on_counter(t)]
        return PerceptionResult(obs["t"], self.perception_mode, targets, dets, (time.perf_counter() - t0) * 1e3)

    @staticmethod
    def _in_bin(t: Target) -> bool:
        return bool(np.all(np.abs(t.pos[:2] - BIN_XY) < BIN_HALF_INNER + 0.01))

    # A few easy drop spots per hand: the object lands just inside the bin's near wall (the G1 palm reaches
    # x ~ 0.46 m at drop height, so the near strip of the bin is what it can drop into without stretching), at
    # three positions across, on the hand's side. Successive drops take the spot furthest from what is
    # already in the bin.
    DROP_ACROSS = (0.03, 0.08, 0.12)   # |y| offsets from the bin centre line

    def choose_drop_spot(self, side: str, t: Target):
        o = OBJECTS[t.name]
        sgn = -1.0 if side == "right" else 1.0
        dx = -BIN_HALF_INNER[0] + max(o.half_x, o.half_y) + 0.035     # bottom edge 3.5 cm inside the near wall
        # (12 mm: 4 of 96 objects bounced back over the wall)
        spots = [BIN_XY + np.array([dx, sgn * dy]) for dy in self.DROP_ACROSS]
        occupied = list(self.in_bin_xy) + list(self.used_spots)

        def clearance(xy):
            return min((float(np.linalg.norm(xy - q)) for q in occupied), default=1.0)
        k = max(range(len(spots)), key=lambda i: (round(clearance(spots[i]), 2), -i))   # ties: list order
        xy = spots[k]
        t.drop_xy = xy
        self.used_spots.append(xy)
        return xy

    @staticmethod
    def _on_counter(t: Target) -> bool:
        x, y = t.pos[:2]
        in_bin = np.all(np.abs(t.pos[:2] - BIN_XY) < np.array([0.15, 0.165]))
        return bool(0.2 < x < 0.8 and abs(y) < 0.55 and not in_bin)

    @staticmethod
    def reachable(t: Target) -> bool:
        """Objects pushed in front of the chest (|y| < 9 cm) or too far cannot be grasped without the wrist
        coming too close to the torso; the planner leaves them (reported as not_attempted) rather than risk it."""
        x, y = t.pos[:2]
        return bool(abs(y) >= 0.09 and 0.28 <= x <= 0.50 and abs(y) <= 0.38)

    def clear_of_bin(self, side: str, t: Target) -> bool:
        """An object lying against the bin cannot be grasped from behind without the fingers hitting the bin
        wall: leave it rather than push the hand into the bin. Checked on the real kinematics: IK for the grasp
        pose, then the open fingertips (+ finger thickness) against the bin's near wall."""
        p = grasp_plan(side, t)
        q, _ = self.ik.solve(self.ik.q_home, {side: (p["grasp"], p["quat"])}, iters=200)
        self.ik.cfg.update(q)
        d = self.ik.cfg.data
        m = self.ik.model
        lo = np.array([BIN_XY[0] - BIN_HALF_INNER[0] - 0.01, BIN_XY[1] - BIN_HALF_INNER[1] - 0.01, 0.0])
        hi = np.array([BIN_XY[0] + BIN_HALF_INNER[0] + 0.01, BIN_XY[1] + BIN_HALF_INNER[1] + 0.01, BIN_WALL_TOP])
        for site in spec.fingertip_sites(side):
            tip = d.site_xpos[m.site(site).id]
            if np.all(tip > lo - BIN_CLEARANCE) and np.all(tip < hi + BIN_CLEARANCE):
                return False
        return True

    # ------------------------------------------------------------------ decision
    def assign_hand(self, t: Target) -> str:
        """The hand whose ready pose is closest to the object takes it."""
        return min(spec.SIDES, key=lambda s: float(np.linalg.norm(t.pos[:2] - self.home_xy[s])))

    def choose(self, targets: list[Target]) -> tuple[str, Target] | None:
        cands = [t for t in targets if self.attempts.get(t.name, 0) < self.max_attempts and self.reachable(t)]
        if not cands:
            return None
        by_side: dict[str, list[Target]] = {s: [] for s in spec.SIDES}
        for t in cands:
            side = self.assign_hand(t)
            if not self.clear_of_bin(side, t):
                t.max_inward_yaw = 0.0          # fingers parallel to the counter edge instead of towards the bin
                if not self.clear_of_bin(side, t):
                    continue
            by_side[side].append(t)
        order = [s for s in spec.SIDES if s != self.last_side] + ([self.last_side] if self.last_side else [])
        for side in order:
            if by_side[side]:
                # outermost first: the back of the hand faces outwards, so taking the inner object first
                # would wedge the hand between two objects
                t = max(by_side[side], key=lambda t: abs(float(t.pos[1])))
                return side, t
        return None
