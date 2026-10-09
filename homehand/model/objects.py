"""Graspable household objects (YCB, CC BY 4.0).

Each object uses its YCB mesh when `homehand fetch-assets` has downloaded it and falls back to a primitive of
the same size otherwise, so the project still runs without the YCB download.

Body frames sit at the centre of the object's footprint, on its bottom face. Sizes are the half extents of the
footprint in the object frame; `box=True` objects are grasped across whichever side keeps the wrist yaw small.
Grasp parameters were tuned with the grasp benchmark (`homehand grasp-bench`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    label: str                   # human readable class name shown in the UI
    ycb_id: str
    mass: float                  # kg (YCB published masses, rounded)
    primitive: tuple             # fallback geom: ("cylinder", r, half_h) | ("box", hx, hy, hz) | ("sphere", r)
    rgba: str
    half_x: float                # footprint half extent along the object x axis
    half_y: float                # footprint half extent along the object y axis
    box: bool = False            # True: yaw matters (grasp aligned with a side); False: round object
    grasp: str = "cylindrical_wrap"  # key of homehand.control.grasps.GRASPS


OBJECTS: dict[str, ObjectSpec] = {
    o.name: o
    for o in [
        ObjectSpec("soup_can", "tomato soup can", "005_tomato_soup_can", 0.35, ("cylinder", 0.034, 0.051),
                   "0.8 0.15 0.1 1", half_x=0.034, half_y=0.034, grasp="cylindrical_wrap"),
        ObjectSpec("mustard", "mustard bottle", "006_mustard_bottle", 0.45, ("box", 0.048, 0.033, 0.095),
                   "0.95 0.8 0.1 1", half_x=0.048, half_y=0.033, box=True, grasp="low_power_wrap"),
        ObjectSpec("sugar_box", "sugar box", "004_sugar_box", 0.35, ("box", 0.0245, 0.047, 0.088),
                   "0.95 0.95 0.9 1", half_x=0.0245, half_y=0.047, box=True, grasp="palmar_box"),
        ObjectSpec("meat_can", "potted meat can", "010_potted_meat_can", 0.37, ("box", 0.05, 0.03, 0.042),
                   "0.2 0.35 0.75 1", half_x=0.05, half_y=0.03, box=True, grasp="tripod_pinch"),
    ]
}
CLASS_NAMES = list(OBJECTS)  # detector class ids follow this order

# ------------------------------------------------------------------ rest poses
# "upright": as in the YCB frame (object z up). "lying": tipped over onto its broadest stable side. For each
# object: the quaternion (wxyz) that turns the upright object into its lying pose (before the yaw about the
# vertical), the object axis that ends up along the table (the long axis the hand wraps around) and the height
# of the lying object.
_S = float(np.sqrt(0.5))
LYING = {
    "soup_can": {"quat": (_S, _S, 0.0, 0.0), "axis": (0.0, 0.0, 1.0), "thickness": 0.068},   # rolls onto its side
    "mustard": {"quat": (_S, _S, 0.0, 0.0), "axis": (0.0, 0.0, 1.0), "thickness": 0.066},    # on its broad face
    "sugar_box": {"quat": (_S, 0.0, _S, 0.0), "axis": (0.0, 0.0, 1.0), "thickness": 0.049},  # flat on the label
    "meat_can": {"quat": (_S, _S, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "thickness": 0.060},   # on its side
}
REST_POSES = ("upright", "lying")


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def lying_dims(name: str, info: dict | None = None) -> dict:
    """Lying object: thickness (vertical), length along its long axis, width across it (m), from the object
    sizes (generated/objects.json `info` when given, else the primitive sizes)."""
    o = OBJECTS[name]
    hx, hy = (info["half_xy"] if info else (o.half_x, o.half_y))
    h = info["height"] if info else 2 * o.primitive[-1]
    dims = np.array([2 * hx, 2 * hy, h])
    q = _rot(LYING[name]["quat"])
    vertical = int(np.argmax(np.abs(q[2])))               # object axis that points up when lying
    long_ = int(np.argmax(np.abs(LYING[name]["axis"])))
    width = 3 - vertical - long_
    return {"thick": float(dims[vertical]), "length": float(dims[long_]), "width": float(dims[width])}


def rest_quat(name: str, rest: str, yaw: float) -> np.ndarray:
    """World orientation of an object resting in `rest`, turned by `yaw` about the vertical. For a lying
    object `yaw` is the heading of its long axis."""
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    if rest == "upright":
        return qz
    ly = LYING[name]
    q = np.asarray(ly["quat"])
    a = _rot(q) @ np.asarray(ly["axis"])                 # long axis after tipping over, before the yaw
    q0 = _qmul(np.array([np.cos(-np.arctan2(a[1], a[0]) / 2), 0, 0, np.sin(-np.arctan2(a[1], a[0]) / 2)]), q)
    return _qmul(qz, q0)                                  # long axis along `yaw`


def rest_of(name: str, quat) -> tuple[str, float]:
    """(rest pose, yaw) of an object from its world orientation: the yaw of the object x axis when upright
    (as before), the heading of the long axis (mod pi) when lying."""
    R = _rot(quat)
    if R[2, 2] > 0.7 or name not in LYING:
        return "upright", float(np.arctan2(R[1, 0], R[0, 0]))
    a = R @ np.asarray(LYING[name]["axis"])
    return "lying", float((np.arctan2(a[1], a[0]) + np.pi / 2) % np.pi - np.pi / 2)


def body_pose_for(name: str, rest: str, x: float, y: float, yaw: float, height: float,
                  table_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Free-joint (pos, quat) that rests the object at (x, y): the body origin is the bottom centre of the
    upright footprint, so a lying object's origin is placed from its geometric centre."""
    q = rest_quat(name, rest, yaw)
    if rest == "upright":
        return np.array([x, y, table_z + 0.0015]), q
    centre = np.array([x, y, table_z + LYING[name]["thickness"] / 2 + 0.0015])
    return centre - _rot(q) @ np.array([0.0, 0.0, height / 2]), q


MAX_HAND_YAW = np.deg2rad(35)  # wrist yaw the arms can hold across the whole workspace


# Wrist yaw the arm can actually hold at a grasp, measured with the IK over the spawn zones (`inward` > 0 turns
# the fingers towards the body midline). The further out the object, the less the hand can turn inwards:
# right hand at |y| = 0.27 m reaches yaw in [-35, 0] deg, at |y| = 0.13 m in [-15, +25] deg (5 deg margin kept).
_YAW_ABS_Y = (0.13, 0.20, 0.27, 0.30)
_YAW_INWARD_MAX = np.deg2rad((20, 10, 0, -8))
_YAW_OUTWARD_MAX = np.deg2rad((12, 25, 30, 30))


def hand_yaw_window(side: str, y: float, max_inward: float | None = None) -> tuple[float, float]:
    """(lo, hi) wrist yaw for `side` grasping an object at lateral position `y` (`max_inward` caps the turn
    towards the midline further, e.g. to keep the fingertips off the bin)."""
    ay = abs(y)
    inward = float(np.interp(ay, _YAW_ABS_Y, _YAW_INWARD_MAX))
    if max_inward is not None:
        inward = min(inward, max_inward)
    outward = float(np.interp(ay, _YAW_ABS_Y, _YAW_OUTWARD_MAX))
    # right hand: inward = +yaw (fingers towards +y); left hand mirrored
    return (-outward, inward) if side == "right" else (-inward, outward)


def grasp_geometry(spec: ObjectSpec, yaw: float, side: str | None = None, y: float | None = None,
                   max_inward: float | None = None) -> tuple[float, float, float]:
    """Return (hand_yaw, extent along the fingers, extent towards the palm) for an object at `yaw`.

    The hand keeps its "handshake" orientation rotated by `hand_yaw`. For boxes either pair of faces can be
    grasped: we take the one whose alignment needs the least wrist yaw outside what the arm can reach (see
    `hand_yaw_window`; without side / position, a symmetric +-35 deg window), then clip to that window; the
    residual misalignment is absorbed by the compliant fingers.
    """
    if not spec.box:
        return 0.0, spec.half_x, spec.half_y
    lo, hi = (-MAX_HAND_YAW, MAX_HAND_YAW) if side is None or y is None else hand_yaw_window(side, y, max_inward)
    a = (yaw + np.pi / 2) % np.pi - np.pi / 2            # object x axis, wrapped to [-90, 90)
    b = a - np.pi / 2 if a >= 0 else a + np.pi / 2       # object y axis, also in [-90, 90)
    cands = [(a, spec.half_x, spec.half_y), (b, spec.half_y, spec.half_x)]

    def misalign(c):
        return abs(c[0] - np.clip(c[0], lo, hi)) + 1e-3 * abs(c[0])   # ties -> smaller wrist yaw
    hy, along, towards = min(cands, key=misalign)
    return float(np.clip(hy, lo, hi)), along, towards
