"""Pick-and-place skill for one hand: a scripted finite-state machine over palm waypoints (mink IK).

The skill only uses proprioception plus a target estimate (from perception), never privileged simulator
state. A failed grasp is detected proprioceptively: if the fingers close all the way, the hand is empty.

Grasp strategy for the Inspire hand (found with the grasp benchmark): "handshake" orientation, thumb
pre-rotated into opposition, the hand approaches *forward* along the finger direction so the object slides
between thumb and fingers without being swept by the thumb; then fingers + thumb close and the arm lifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from homehand.control.grasps import GRASPS, LYING_GRASP, RELAXED_HAND
from homehand.control.ik import ArmIK
from homehand.control.safety import ARM_MAX_VEL
from homehand.env.kitchen_env import BIN_WALL_TOP, BIN_XY, TABLE_Z
from homehand.model import spec
from homehand.model.objects import OBJECTS, grasp_geometry, lying_dims

REACH_SPEED = 0.35          # m/s, peak palm speed of a free-space move
FINE_SPEED = 0.15
CORNER_RADIUS = 0.06        # m, via points are passed on a rounded corner instead of stopping there
APPROACH_BACKOFF = 0.07     # the hand starts this far behind the grasp pose and moves forward
DROP_OFFSET_Y = 0.07        # each hand drops into its own half of the bin
DROP_CLEARANCE = 0.045      # object bottom above the bin rim at release (fingers stay clear of the rim)
SAFE_Z = 1.10               # palm height for lateral moves while carrying: a held object clears the tallest one
KEEP_YAW = True
REACH_Z = 1.03              # palm height above the pre-grasp point (tallest object top: 0.99 m; higher is out of reach near the chest)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def rot2(yaw: float, v) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass
class Target:
    """What a skill needs to know about the object: class + footprint pose (from perception or ground truth)."""
    name: str
    pos: np.ndarray        # footprint centre (x, y, z_table)
    yaw: float
    height: float
    max_inward_yaw: float | None = None   # set by the planner when the default wrist yaw would hit the bin
    drop_xy: np.ndarray | None = None     # where the object should land in the bin (chosen by the planner)
    rest: str = "upright"                 # "upright" | "lying" (then `yaw` is the heading of its long axis)

    def to_dict(self) -> dict:
        return {"name": self.name, "pos": [round(float(v), 4) for v in self.pos], "yaw": round(self.yaw, 3),
                "height": round(self.height, 4), "rest": self.rest}


def min_jerk(tau: float) -> float:
    """Minimum-jerk time scaling: zero velocity and acceleration at both ends (human reaching profile)."""
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


MIN_JERK_PEAK = 1.875   # peak / mean speed of a minimum-jerk profile
JOINT_SPEED_FRAC = 0.9  # planned peak joint speed as a fraction of the safety limit
WRIST_TURN_SPEED = 1.5  # rad/s, peak palm rotation speed
PATH_STEP = 0.02        # m between IK samples of a palm path


def rounded_path(points: list[np.ndarray], radius: float = CORNER_RADIUS, n_arc: int = 12) -> np.ndarray:
    """Dense polyline through `points` whose interior corners are replaced by quadratic Bezier arcs.

    The first and last points are hit exactly, the via points are cut by at most `radius` (and never by more
    than 40 % of the adjacent legs, so a short vertical leg such as the final descent stays vertical)."""
    pts = [np.asarray(p, float) for p in points]
    pts = [p for i, p in enumerate(pts) if i == 0 or np.linalg.norm(p - pts[i - 1]) > 1e-6]
    if len(pts) < 3:
        return np.array(pts if len(pts) > 1 else pts * 2)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        l1, l2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
        r = min(radius, 0.4 * l1, 0.4 * l2)
        p0, p2 = b + (a - b) * r / l1, b + (c - b) * r / l2
        for t in np.linspace(0.0, 1.0, n_arc):
            out.append((1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t ** 2 * p2)
    out.append(pts[-1])
    return np.array(out)


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(dot)
    return (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)


@dataclass
class Segment:
    """One motion primitive: the palm follows a rounded path start -> via... -> goal with a minimum-jerk
    profile (peak speed `speed`), the orientation is slerped and the hand command blends along with it."""
    name: str
    goal: Callable[[], tuple[np.ndarray, np.ndarray]]  # -> palm (pos, quat)
    hand: tuple | None = None   # (thumb_yaw, thumb_pitch, index, middle, ring, pinky) at the end of the segment
    speed: float = REACH_SPEED
    min_steps: int = 1
    hold_steps: int = 0
    on_end: Callable[[], None] | None = None
    via: Callable[[], list] | None = None   # intermediate palm positions passed without stopping
    joint_space: bool = False   # free-space move: interpolate joints between IK key poses (reach / return)
    final_q: Callable[[], np.ndarray] | None = None   # joint_space: end in this configuration


TOP_APPROACH_H = 0.10      # the palm comes down vertically from this far above its grasp pose
TOP_SWING_Z = 1.12        # palm height while swinging over to a lying object
RETURN_Z = 1.22           # palm height above the ready pose before the arm folds down into it
TOP_DROP_TILT = np.deg2rad(10)
TOP_DROP_CLEARANCE = 0.08  # higher than side drops: the palm-down forearm slopes down behind the hand and would
                           # otherwise pass just above the tall objects in front of the body   # hand orientation over the bin for top grasps (reachable, see top_grasp_plan)
TOP_DROP_YAW = np.deg2rad(15)
# Palm-down wrist yaw the arm reaches best (measured with the IK, right hand; the left mirrors): fingers pointing
# ~15 deg inwards for the inner slot, ~20 deg outwards for the outer one; reachable within about +-40 deg.
_TOP_PREF_Y = (0.13, 0.27)
_TOP_PREF_YAW = np.deg2rad((15.0, -20.0))


def palm_down_quat(side: str, yaw: float) -> np.ndarray:
    """Palm-site orientation with the palm facing down, fingers along `yaw`, thumb towards the midline."""
    a = -np.pi / 2 if side == "right" else np.pi / 2
    qx = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])
    qz = yaw_quat(yaw)
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qx
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def top_hand_quat(side: str, yaw: float, tilt: float) -> np.ndarray:
    """Palm down, fingers along `yaw`, then tilted fingers-down by `tilt` about the thumb axis."""
    tq = np.array([np.cos(tilt / 2), 0.0, 0.0, np.sin((tilt if side == "right" else -tilt) / 2)])
    return _qmul(palm_down_quat(side, yaw), tq)


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qrot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def top_grasp_plan(side: str, target: Target) -> dict:
    """Grasp of a lying object from above (palm down, fingers tilted down; per-object geometry in grasps.py,
    found with scripts/top_grasp_search.py). The hand comes down vertically onto the object and closes."""
    gt = GRASPS[LYING_GRASP[target.name]]
    dims = lying_dims(target.name)
    thumb_heading = target.yaw if gt.align == "along" else target.yaw + np.pi / 2
    # fingers perpendicular to the thumb, pointing away from the robot ((-90, 90] deg): the only way it reaches
    yaw = float((thumb_heading - np.pi / 2 + np.pi / 2) % np.pi - np.pi / 2)
    q = top_hand_quat(side, yaw, np.deg2rad(gt.tilt_deg))
    centre = np.array([target.pos[0], target.pos[1], TABLE_Z + dims["thick"] / 2])
    dx, dy, dz = gt.obj_in_palm
    grasp = centre - _qrot(q) @ np.array([dx, dy if side == "right" else -dy, dz])   # left palm normal = -y
    sgn_y = -1.0 if side == "right" else 1.0
    obj_drop = target.drop_xy if target.drop_xy is not None else BIN_XY + np.array([-0.03, sgn_y * DROP_OFFSET_Y])
    # drop: where the palm must be for the object, held where it really sits in the closed hand, to be centred
    # over the drop spot with its bottom DROP_CLEARANCE above the rim
    # The steep grasp pose cannot reach over the bin: while carrying, the wrist turns the hand nearly flat
    # (palm down, fingers 10 deg down) and 15 deg inwards, which the arm reaches at the drop height.
    sgn_in = 1.0 if side == "right" else -1.0
    dq = top_hand_quat(side, sgn_in * TOP_DROP_YAW, TOP_DROP_TILT)
    hx, hy, hz = gt.held_in_palm or gt.obj_in_palm
    held = _qrot(dq) @ np.array([hx, hy if side == "right" else -hy, hz])
    obj_c = np.array([*np.asarray(obj_drop, float), BIN_WALL_TOP + TOP_DROP_CLEARANCE + dims["thick"] / 2])
    drop = obj_c - held
    return {"approach": "top", "hand_yaw": yaw, "quat": q, "grasp": grasp,
            "pre": grasp + np.array([0.0, 0.0, TOP_APPROACH_H]), "back": np.zeros(2),
            "drop": drop, "drop_quat": dq, "grasp_type": gt}


def grasp_plan(side: str, target: Target) -> dict:
    """Palm poses for grasping `target` with `side` (shared by the scripted skill and dataset features)."""
    if target.rest == "lying":
        return top_grasp_plan(side, target)
    o = OBJECTS[target.name]
    gt = GRASPS[o.grasp]
    hand_yaw, along, towards = grasp_geometry(o, target.yaw, side, float(target.pos[1]), target.max_inward_yaw)
    sgn = -1.0 if side == "right" else 1.0
    offset = rot2(hand_yaw, (-(along + gt.reach), sgn * (towards + gt.palm_gap)))
    grip_h = max(gt.min_z, gt.z_frac * target.height)
    palm_xy = target.pos[:2] + offset
    # the wrist turns back to yaw 0 while carrying (easier to reach over the bin)
    drop_offset = offset if KEEP_YAW else np.array([-(along + gt.reach), sgn * (towards + gt.palm_gap)])
    obj_drop = target.drop_xy if target.drop_xy is not None else BIN_XY + np.array([-0.03, sgn * DROP_OFFSET_Y])
    drop_xy = np.asarray(obj_drop, float) + drop_offset
    return {
        "approach": "side",
        "hand_yaw": hand_yaw,
        "quat": yaw_quat(hand_yaw),
        "grasp": np.array([palm_xy[0], palm_xy[1], TABLE_Z + grip_h]),
        "back": rot2(hand_yaw, (-APPROACH_BACKOFF, 0.0)),
        # release point: the object's bottom 3 cm above the bin rim, it is dropped (not placed) into the bin
        "drop": np.array([drop_xy[0], drop_xy[1], BIN_WALL_TOP + DROP_CLEARANCE + grip_h]),
        "drop_quat": yaw_quat(hand_yaw if KEEP_YAW else 0.0),
        "grasp_type": gt,
    }


class PickPlaceSkill:
    """Scripted expert for one pick-and-place. Call `act(obs)` every control step until `done`."""

    def __init__(self, ik: ArmIK, side: str, target: Target, q_start: np.ndarray, hand_cmd: dict):
        self.ik, self.side, self.target = ik, side, target
        self.q = q_start.copy()
        self.hand_cmd = {s: np.asarray(v, float).copy() for s, v in hand_cmd.items()}
        self.status = "running"  # -> "placed" | "empty_grasp"
        self.steps = 0
        p = grasp_plan(side, target)
        self.plan_info = p
        quat, dq, gt = p["quat"], p["drop_quat"], p["grasp_type"]
        self.grasp_type = gt
        g, back = p["grasp"], np.array([*p["back"], 0.0])
        open_, closed = gt.open_cmd(), gt.closed_cmd()
        relaxed = RELAXED_HAND
        home_pos, home_quat = ik.palm_pose(ik.q_home, side)
        # the arm comes back high and then folds down into the ready pose: the forearm stays above the objects
        # in front of the body (it swept the tall ones when folding straight from the bin)
        above_home = np.array([home_pos[0], home_pos[1], RETURN_Z])

        def fixed(pos, q=quat):
            return lambda: (np.asarray(pos, float), q)

        def here():
            return self.ik.palm_pose(self.q, side)

        def high(pos):
            return np.array([pos[0], pos[1], SAFE_Z])

        def check_grasp():
            # (top grasps close the fingers deep around the object even when they hold it: no proprioceptive
            # check there; a missed object is still on the counter at the next look and gets another attempt)
            if gt.approach == "side" and self.finger_flexion() > gt.empty_threshold():
                self.status = "empty_grasp"
                self.plan[1:] = [
                    Segment("open", here, open_, min_steps=8),
                    Segment("back_off", fixed(p.get("pre", g + back)), open_, speed=FINE_SPEED, min_steps=8),
                    Segment("return", lambda: (home_pos, home_quat), relaxed, min_steps=15,
                            via=lambda: [high(g + back)], joint_space=True, final_q=lambda: ik.q_home),
                ]

        def placed():
            self.status = "placed"

        # Lateral moves happen at SAFE_Z; the hand only goes down vertically behind the object (fingertips
        # stay in front of the robot, away from the other objects) and approaches it forward. Free-space moves
        # pass their via points on rounded corners (one fluid reach / carry / return motion instead of
        # stop-and-go), every move starts and ends with zero velocity and acceleration (minimum jerk).
        drop = p["drop"]
        if p["approach"] == "top":
            pre = p["pre"]
            self.plan = [
                # swing the arm over the object, palm down, thumb out of the way; come down vertically onto it
                # (the tilted fingers and the opposed thumb hang ~10 cm below the palm: swing over at TOP_SWING_Z so
                # they clear the tallest neighbour, then come down vertically)
                Segment("reach", fixed([pre[0], pre[1], max(pre[2], TOP_SWING_Z)]), open_, min_steps=20,
                        joint_space=True),
                Segment("descend", fixed(pre), open_, speed=0.25, min_steps=8),
                Segment("approach", fixed(g), open_, speed=FINE_SPEED, min_steps=12, hold_steps=2),
                Segment("grasp", fixed(g), closed, min_steps=18, hold_steps=6, on_end=check_grasp),
                Segment("lift", fixed(high(g)), closed, speed=0.2, min_steps=14),
                Segment("transport", fixed(drop, dq), closed, speed=0.3, min_steps=24, hold_steps=2,
                        via=lambda: [high(drop)], joint_space=True),
                Segment("drop", fixed(drop, dq), open_, min_steps=6, hold_steps=6, on_end=placed),
                Segment("return", lambda: (home_pos, home_quat), relaxed, min_steps=20,
                        via=lambda: [drop + [0, 0, 0.08], [drop[0] - 0.06, drop[1], RETURN_Z], above_home],
                        joint_space=True, final_q=lambda: ik.q_home),
            ]
            self._start()
            return
        self.plan: list[Segment] = [
            # swing the arm (joint space) to above the pre-grasp point, then come down vertically behind the object
            Segment("reach", fixed([*(g + back)[:2], REACH_Z]), open_, min_steps=20, joint_space=True),
            Segment("descend", fixed(g + back), open_, speed=0.25, min_steps=8),
            Segment("approach", fixed(g), open_, speed=FINE_SPEED, min_steps=10, hold_steps=2),
            Segment("grasp", fixed(g), closed, min_steps=15, hold_steps=5, on_end=check_grasp),
            # lift straight up (the object must not drag over the counter), then swing over to the bin and come
            # down above it: a joint-space move through the key poses, on the IK's well-conditioned branch
            Segment("lift", fixed(high(g)), closed, speed=0.25, min_steps=12),
            Segment("transport", fixed(drop, dq), closed, speed=0.3, min_steps=24, hold_steps=2,
                    via=lambda: [high(drop)], joint_space=True),
            Segment("drop", fixed(drop, dq), open_, min_steps=6, hold_steps=6, on_end=placed),
            # straight up out of the bin first, then a smooth arc back to the ready pose
            Segment("return", lambda: (home_pos, home_quat), relaxed, min_steps=20,
                    via=lambda: [drop + [0, 0, 0.08], [drop[0] - 0.06, drop[1], RETURN_Z], above_home], joint_space=True,
                    final_q=lambda: ik.q_home),
        ]
        self._start()

    # -------------------------------------------------------------------------------------------
    def finger_flexion(self) -> float:
        """Mean flexion of index + middle (they close in every grasp type)."""
        hq = self._last_obs["hand_q"]
        base = 0 if self.side == "left" else 6
        return float(np.mean(hq[base + 2: base + 4]))

    @property
    def done(self) -> bool:
        return not self.plan

    @property
    def phase(self) -> str:
        return self.plan[0].name if self.plan else "done"

    def _start(self) -> None:
        if not self.plan:
            return
        seg = self.plan[0]
        self.start_pos, self.start_quat = self.ik.palm_pose(self.q, self.side)
        goal, self.goal_quat = seg.goal()
        via = [np.asarray(v, float) for v in seg.via()] if seg.via is not None else []
        self.path = rounded_path([self.start_pos, *via, goal])
        seglen = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.arc = np.r_[0.0, np.cumsum(seglen)]
        # Joint-space plan: sample the palm path densely, solve IK for every sample, then time the motion.
        # Each sample gets the time its slowest constraint needs (palm speed, joint speed, wrist rotation), and
        # a minimum-jerk profile runs over that cumulative "time-at-full-speed": smooth start / stop, no joint
        # ever above JOINT_SPEED_FRAC of its limit, and no IK runs while the arm moves (no twitching).
        angle = 2 * np.arccos(min(1.0, abs(float(np.dot(self.start_quat, self.goal_quat)))))
        n = int(max(1, np.ceil(self.arc[-1] / PATH_STEP), np.ceil(angle / 0.05)))
        ss = np.linspace(0.0, 1.0, n + 1)[1:]
        # the palm finishes turning by the last via point: it then comes down onto the object (or into the
        # ready pose) already in its final orientation, so a tilted hand never sweeps over an object's top
        turn_end = 1.0
        if via and self.arc[-1] > 1e-6:
            i = int(np.argmin(np.linalg.norm(self.path - via[-1], axis=1)))
            turn_end = max(self.arc[i] / self.arc[-1], 0.2)
        idx = self.ik.arm_qadr[self.side]
        if seg.joint_space:
            keys = [(np.asarray(v, float), self.goal_quat) for v in via] + [(np.asarray(goal, float), self.goal_quat)]
            self.q_path = self.ik.plan_joint_path(self.q, self.side, keys,
                                                  final_q=seg.final_q() if seg.final_q else None)
            palms = np.array([self.ik.palm_pose(q, self.side)[0] for q in self.q_path])
            dp = np.linalg.norm(np.diff(palms, axis=0), axis=1)
            n = len(dp)
            turn = np.full(n, angle / n)
        else:
            poses = [(self._path_point(x), slerp(self.start_quat, self.goal_quat, min(1.0, x / turn_end))) for x in ss]
            self.q_path = self.ik.plan_path(self.q, self.side, poses)
            dp = np.diff(np.r_[0.0, ss]) * self.arc[-1]
            turn = np.where(ss <= turn_end + 1e-9, angle / max(1, int(np.sum(ss <= turn_end + 1e-9))), 0.0)
        dq = np.abs(np.diff(self.q_path[:, idx], axis=0)).max(axis=1)
        cost = np.maximum.reduce([dp / seg.speed, dq / (JOINT_SPEED_FRAC * ARM_MAX_VEL), turn / WRIST_TURN_SPEED])
        self.u = np.r_[0.0, np.cumsum(cost)]
        self.n_steps = max(seg.min_steps, int(np.ceil(MIN_JERK_PEAK * self.u[-1] / spec.CONTROL_DT)))
        self.k = 0
        self.hand_start = self.hand_cmd[self.side].copy()

    def _joint_point(self, alpha: float) -> np.ndarray:
        u = alpha * self.u[-1]
        i = int(np.clip(np.searchsorted(self.u, u, side="right") - 1, 0, len(self.u) - 2))
        w = 0.0 if self.u[i + 1] <= self.u[i] else (u - self.u[i]) / (self.u[i + 1] - self.u[i])
        return self.q_path[i] + (self.q_path[i + 1] - self.q_path[i]) * min(max(w, 0.0), 1.0)

    def _path_point(self, s: float) -> np.ndarray:
        if self.arc[-1] < 1e-9:
            return self.path[-1].copy()
        d = s * self.arc[-1]
        return np.array([np.interp(d, self.arc, self.path[:, i]) for i in range(3)])

    def act(self, obs: dict) -> np.ndarray:
        self._last_obs = obs
        self.steps += 1
        if self.plan:
            seg = self.plan[0]
            self.k += 1
            alpha = min_jerk(self.k / self.n_steps)
            self.q = self._joint_point(alpha)
            if seg.hand is not None:
                self.hand_cmd[self.side] = self.hand_start + (np.asarray(seg.hand) - self.hand_start) * alpha
            if self.k >= self.n_steps + seg.hold_steps:
                if seg.on_end is not None:
                    seg.on_end()
                self.plan.pop(0)
                self._start()
        return compose_action(self.ik, self.q, self.hand_cmd)


def compose_action(ik: ArmIK, q: np.ndarray, hand_cmd: dict) -> np.ndarray:
    a = np.zeros(spec.ACTION_DIM)
    a[:14] = ik.arm_targets(q)
    for i, s in enumerate(["left", "right"]):
        a[14 + 6 * i: 20 + 6 * i] = hand_cmd[s]
    return a
