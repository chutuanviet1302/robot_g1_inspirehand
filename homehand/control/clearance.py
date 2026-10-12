"""Clearance between the moving arm and the *other* objects on the counter (from perception).

Failure analysis of 50 expert episodes (ground-truth poses) showed that 8 of 11 lost objects were not the one
being picked: the wrist / forearm sloping down behind the hand over the bin touched the top of a tall box in the
inner slot, or a fingertip brushed a neighbour during the approach. Every collision geom of the arm from the
elbow down (forearm, wrist links, palm, every finger link) is approximated by a capsule fitted to its bounding box
(a row of spheres), the object being carried by one more capsule fixed in the palm frame, and every object on the
counter by its oriented bounding box (grown by the perception error).

HOMEHAND_PROBES=v1 restores the original probes (spheres at the wrist links, palm and fingertips only).
"""

from __future__ import annotations

import os

import numpy as np

from homehand.env.kitchen_env import TABLE_Z
from homehand.model import spec
from homehand.model.objects import OBJECTS, lying_dims

PROBES = os.environ.get("HOMEHAND_PROBES", "capsules")
WRIST_LINKS = ("wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
WRIST_RADIUS = 0.035     # m, around the wrist link origins (v1 probes)
FINGER_RADIUS = 0.010
MARGIN = 0.015
# how much the obstacle boxes grow per perception mode (YOLO position error: mean 3.5 mm, p90 5.4 mm)
PERCEPTION_INFLATE = {"ground_truth": 0.0, "oracle": 0.004, "yolo": 0.006}


def obstacle_boxes(targets, inflate: float = 0.0) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(centre, rotation (3x3, columns = box axes), half extents) of each object."""
    out = []
    for t in targets:
        c, s = np.cos(t.yaw), np.sin(t.yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        if t.rest == "lying":
            d = lying_dims(t.name)
            half = np.array([d["length"] / 2, d["width"] / 2, d["thick"] / 2])
            h = d["thick"]
        else:
            o = OBJECTS[t.name]
            half = np.array([o.half_x, o.half_y, t.height / 2])
            h = t.height
        out.append((np.array([t.pos[0], t.pos[1], TABLE_Z + h / 2]), R, half + inflate))
    return out


def capsule_points(centre: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, float]:
    """Spheres (local centres, common radius) along the longest axis of a box: a capsule that covers its
    cross-section (radius = larger of the two short half extents), one sphere per radius of length."""
    ax = int(np.argmax(half))
    r = float(max(np.delete(half, ax)))
    reach = max(float(half[ax]) - r, 0.0)
    n = int(np.ceil(2 * reach / max(r, 1e-3))) + 1 if reach > 0 else 1
    pts = np.tile(np.asarray(centre, float), (n, 1))
    pts[:, ax] += np.linspace(-reach, reach, n)
    return pts, r


class ArmProbes:
    """Capsules of one arm's collision geoms from the elbow down, in their geom frames (computed once)."""

    def __init__(self, model, side: str):
        prefix = spec.PREFIX[side]
        bodies = {model.body(f"{side}_elbow_link").id} | \
            {model.body(f"{side}_{link}").id for link in WRIST_LINKS} | \
            {b for b in range(model.nbody) if model.body(b).name.startswith(prefix)}
        self.geoms, self.local, self.radius = [], [], []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] in bodies and (model.geom_contype[g] or model.geom_conaffinity[g]):
                pts, r = capsule_points(model.geom_aabb[g, :3], model.geom_aabb[g, 3:])
                self.geoms += [g] * len(pts)
                self.local.append(pts)
                self.radius += [r] * len(pts)
        self.geoms = np.array(self.geoms)
        self.local = np.vstack(self.local)
        self.radius = np.array(self.radius)

    def world(self, data) -> np.ndarray:
        R = data.geom_xmat[self.geoms].reshape(-1, 3, 3)
        return data.geom_xpos[self.geoms] + np.einsum("nij,nj->ni", R, self.local)


_ARM_PROBES: dict = {}


def arm_probes(ik, side: str) -> ArmProbes:
    key = (id(ik.model), side)
    if key not in _ARM_PROBES:
        _ARM_PROBES[key] = ArmProbes(ik.model, side)
    return _ARM_PROBES[key]


class HandJoints:
    """Where the 6 hand commands (+ their mimic joints) go in the IK model's qpos: the IK keeps the fingers
    frozen at 0 (straight, thumb not rotated), the real hand is open / pre-shaped / closed."""

    def __init__(self, model, side: str):
        self.act = np.array([model.jnt_qposadr[model.joint(j).id] for j in spec.hand_actuators(side)])
        self.mimic = []
        for i in range(model.neq):
            if model.eq_type[i] == 2:     # mjEQ_JOINT: q1 = c0 + c1 q2 (+ ...)
                j1, j2 = model.eq_obj1id[i], model.eq_obj2id[i]
                if model.joint(j1).name.startswith(spec.PREFIX[side]):
                    self.mimic.append((model.jnt_qposadr[j1], model.jnt_qposadr[j2], *model.eq_data[i][:2]))

    def apply(self, q: np.ndarray, cmd) -> np.ndarray:
        q = q.copy()
        q[self.act] = cmd
        for a1, a2, c0, c1 in self.mimic:
            q[a1] = c0 + c1 * q[a2]
        return q


_HAND_JOINTS: dict = {}


def with_hand(ik, q: np.ndarray, side: str, cmd) -> np.ndarray:
    """q with the hand of `side` in the commanded shape (cmd = 6 hand targets, None = leave as is)."""
    if cmd is None or PROBES == "v1":
        return q
    key = (id(ik.model), side)
    if key not in _HAND_JOINTS:
        _HAND_JOINTS[key] = HandJoints(ik.model, side)
    return _HAND_JOINTS[key].apply(q, np.asarray(cmd, float))


def held_capsule(box, palm_pos: np.ndarray, palm_R: np.ndarray) -> tuple[np.ndarray, float]:
    """The carried object (its box at the grasp) as a capsule in the palm frame of the grasp pose."""
    c, R, half = box
    pts, r = capsule_points(np.zeros(3), half)
    world = c + pts @ R.T
    return (world - palm_pos) @ palm_R, r


def probes(ik, q: np.ndarray, side: str, held=None) -> tuple[np.ndarray, np.ndarray]:
    """Probe spheres of one arm + hand (+ carried object) in configuration q: centres (n, 3), radii (n,)."""
    ik.cfg.update(q)
    d, m = ik.cfg.data, ik.model
    if PROBES == "v1":
        pts = [d.xpos[m.body(f"{side}_{link}").id] for link in WRIST_LINKS]
        rad = [WRIST_RADIUS] * len(pts)
        tips = [d.site_xpos[m.site(s).id] for s in (*spec.fingertip_sites(side), spec.palm_site(side))]
        return np.array(pts + tips), np.array(rad + [FINGER_RADIUS] * len(tips))
    ap = arm_probes(ik, side)
    pts, rad = ap.world(d), ap.radius
    if held is not None:
        local, r = held
        sid = m.site(spec.palm_site(side)).id
        hp = d.site_xpos[sid] + local @ d.site_xmat[sid].reshape(3, 3).T
        pts, rad = np.vstack([pts, hp]), np.r_[rad, np.full(len(hp), r)]
    return pts, rad


TOP_SLACK = 0.01   # m: probes this far below a "top" box's top face still count as coming down onto it


def box_distance(p: np.ndarray, box) -> np.ndarray | float:
    """Distance from point(s) to an oriented box (negative inside)."""
    c, R, half = box[:3]
    local = (np.atleast_2d(p) - c) @ R
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    out = np.where(outside > 0, outside, q.max(axis=1))
    return float(out[0]) if np.ndim(p) == 1 else out


def clearances(ik, q_path, side: str, boxes, stride: int = 2, held=None, hands=None) -> np.ndarray:
    """Smallest gap (m) between the arm probes and each box along a joint path (negative = collision).
    `hands`: the hand command along the path (one 6-vector for all of it, or one per configuration)."""
    best = np.full(len(boxes), np.inf)
    if not boxes:
        return best
    q_path = np.asarray(q_path)
    if hands is not None and np.ndim(hands) == 1:
        hands = np.tile(np.asarray(hands, float), (len(q_path), 1))
    for k in range(0, len(q_path), stride):
        q = with_hand(ik, q_path[k], side, None if hands is None else hands[k])
        pts, rad = probes(ik, q, side, held)
        for i, b in enumerate(boxes):
            if len(b) == 4 and b[3] == "top":
                # only what comes down onto the box counts (probes above its top face): the open fingers of a
                # side grasp end up level with the object, beside it, and then slide around it
                keep = pts[:, 2] > b[0][2] + b[2][2] - TOP_SLACK
                if not keep.any():
                    continue
                best[i] = min(best[i], float((box_distance(pts[keep], b[:3]) - rad[keep]).min()))
            else:
                best[i] = min(best[i], float((box_distance(pts, b) - rad).min()))
    return best


def min_clearance(ik, q_path, side: str, boxes, stride: int = 2, held=None, hands=None) -> float:
    """Smallest gap (m) between the arm probes and the boxes along a joint path (negative = collision)."""
    return float(clearances(ik, q_path, side, boxes, stride, held, hands).min(initial=np.inf))
