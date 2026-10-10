"""Clearance between the moving arm and the *other* objects on the counter (from perception).

Failure analysis of 50 expert episodes (ground-truth poses) showed that 8 of 11 lost objects were not the one
being picked: the wrist / forearm sloping down behind the hand over the bin touched the top of a tall box in the
inner slot, or a fingertip brushed a neighbour during the approach. The arm is approximated by probe spheres
(wrist links, palm, fingertips) and every object by its oriented bounding box.
"""

from __future__ import annotations

import numpy as np

from homehand.env.kitchen_env import TABLE_Z
from homehand.model import spec
from homehand.model.objects import OBJECTS, lying_dims

WRIST_LINKS = ("wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
WRIST_RADIUS = 0.035     # m, around the wrist link origins (the links are ~6 cm thick)
FINGER_RADIUS = 0.010
MARGIN = 0.015


def obstacle_boxes(targets) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        out.append((np.array([t.pos[0], t.pos[1], TABLE_Z + h / 2]), R, half))
    return out


def probes(ik, q: np.ndarray, side: str) -> list[tuple[np.ndarray, float]]:
    """Probe spheres (centre, radius) of one arm + hand in configuration q."""
    ik.cfg.update(q)
    d, m = ik.cfg.data, ik.model
    pts = [(d.xpos[m.body(f"{side}_{link}").id].copy(), WRIST_RADIUS) for link in WRIST_LINKS]
    pts += [(d.site_xpos[m.site(s).id].copy(), FINGER_RADIUS) for s in spec.fingertip_sites(side)]
    pts.append((d.site_xpos[m.site(spec.palm_site(side)).id].copy(), FINGER_RADIUS))
    return pts


def box_distance(p: np.ndarray, box) -> float:
    """Distance from a point to an oriented box (negative inside)."""
    c, R, half = box
    local = R.T @ (p - c)
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    return float(outside if outside > 0 else np.max(q))


def min_clearance(ik, q_path, side: str, boxes, stride: int = 2) -> float:
    """Smallest gap (m) between the arm probes and the boxes along a joint path (negative = collision)."""
    best = np.inf
    if not boxes:
        return best
    for q in q_path[::stride]:
        for p, r in probes(ik, q, side):
            for b in boxes:
                best = min(best, box_distance(p, b) - r)
    return best
