"""2D detection + depth -> 3D object estimate (footprint centre, height, yaw).

The head camera looks down at the counter, so for upright objects it sees the top face (whose far edge
bounds the object away from the camera) and the front face (which bounds it towards the camera). The
footprint centre is therefore the midpoint of the visible points' extent along the object's principal axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from homehand.env.kitchen_env import TABLE_Z
from homehand.model.objects import LYING, OBJECTS, lying_dims
from homehand.perception.camera import Frame, deproject


@dataclass
class Detection:
    name: str                    # class name (key of OBJECTS)
    score: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    mask: np.ndarray | None = None   # (H, W) bool


@dataclass
class ObjectEstimate:
    name: str
    score: float
    pos: np.ndarray              # footprint centre on the table (x, y, TABLE_Z)
    height: float
    yaw: float                   # yaw of the object x axis (mod pi for boxes, 0 for round objects); lying: long axis
    n_points: int
    bbox: tuple[int, int, int, int]
    rest: str = "upright"        # "upright" | "lying" (from the measured height)

    def to_dict(self) -> dict:
        return {"name": self.name, "label": OBJECTS[self.name].label, "score": round(self.score, 3),
                "pos": [round(float(v), 4) for v in self.pos], "height": round(self.height, 4),
                "yaw": round(self.yaw, 3), "rest": self.rest, "bbox": list(map(int, self.bbox))}


def localize(det: Detection, frame: Frame, min_points: int = 40) -> ObjectEstimate | None:
    x1, y1, x2, y2 = det.bbox
    if det.mask is not None:
        vs, us = np.nonzero(det.mask)
    else:
        vv, uu = np.mgrid[y1:y2, x1:x2]
        vs, us = vv.ravel(), uu.ravel()
    if len(us) < min_points:
        return None
    pts = deproject(frame, us, vs)
    pts = pts[pts[:, 2] > TABLE_Z + 0.008]       # drop table pixels leaking into the mask
    if len(pts) < min_points:
        return None
    if det.mask is None:                          # bbox only: keep the nearest depth cluster
        d = np.linalg.norm(pts - frame.cam_pos, axis=1)
        pts = pts[d < np.percentile(d, 10) + 0.12]
    spec = OBJECTS[det.name]
    xy = pts[:, :2]
    if len(xy) > 4000:  # plenty for a footprint fit
        xy = xy[np.random.default_rng(0).choice(len(xy), 4000, replace=False)]
    view = xy.mean(0) - frame.cam_pos[:2]
    view /= np.linalg.norm(view) + 1e-9
    top = float(np.percentile(pts[:, 2], 99))
    if det.name in LYING:
        # Rest pose from the height of the visible top: a tipped-over object is far lower than an upright one
        # (soup can 6.8 vs 10.2 cm, mustard 6.6 vs 19, sugar box 4.9 vs 17.6, meat can 6.0 vs 8.4 cm).
        ld = lying_dims(det.name)
        upright_h = 2 * spec.primitive[-1]
        if top - TABLE_Z < 0.5 * (upright_h + ld["thick"]):
            th, centre = _fit_rectangle(xy, ld["length"] / 2, ld["width"] / 2, view)
            return ObjectEstimate(det.name, det.score, np.array([centre[0], centre[1], TABLE_Z]), ld["thick"],
                                  float((th + np.pi / 2) % np.pi - np.pi / 2), len(pts), det.bbox, rest="lying")
    if spec.box:
        yaw, centre = _fit_rectangle(xy, spec.half_x, spec.half_y, view)
    else:
        yaw = 0.0
        centre = _fit_footprint(xy, view, np.array([-view[1], view[0]]), spec.half_x, spec.half_x, view)
    return ObjectEstimate(det.name, det.score, np.array([centre[0], centre[1], TABLE_Z]), top - TABLE_Z,
                          float((yaw + np.pi / 2) % np.pi - np.pi / 2), len(pts), det.bbox)


def _fit_footprint(xy, a, b, ha, hb, view) -> np.ndarray:
    """Centre of a (2ha x 2hb) footprint with axes a, b containing the visible points.

    Along each axis, if the visible extent is shorter than the true size, the missing part is on the side
    facing away from the camera (occluded by the object itself), so we anchor on the camera-facing edge.
    """
    centre = np.zeros(2)
    for axis, h in ((a, ha), (b, hb)):
        p = xy @ axis
        lo, hi = np.percentile(p, [1, 99])
        if hi - lo >= 2 * h:
            c = (lo + hi) / 2
        else:
            c = lo + h if axis @ view > 0 else hi - h
            c = 0.5 * c + 0.5 * np.clip((lo + hi) / 2, hi - h, lo + h) if abs(axis @ view) < 0.5 else c
        centre += axis * c
    return centre


def _fit_rectangle(xy, hx, hy, view) -> tuple[float, np.ndarray]:
    """Yaw of the object x axis + footprint centre for a box of known half extents (hx, hy).

    Search the orientation whose bounding rectangle is tightest while still matching the known size;
    the head camera sees the full top face of a box, so the tightest rectangle is the footprint itself.
    """
    best = (np.inf, 0.0)
    for th in np.deg2rad(np.arange(0.0, 180.0, 2.0)):
        a = np.array([np.cos(th), np.sin(th)])
        b = np.array([-a[1], a[0]])
        pa, pb = xy @ a, xy @ b
        ea = np.subtract(*np.percentile(pa, [99, 1]))
        eb = np.subtract(*np.percentile(pb, [99, 1]))
        overflow = max(0.0, ea - 2 * hx) + max(0.0, eb - 2 * hy)
        cost = 10 * overflow + ea * eb
        if cost < best[0]:
            best = (cost, th)
    th = best[1]
    a = np.array([np.cos(th), np.sin(th)])
    b = np.array([-a[1], a[0]])
    return float(th), _fit_footprint(xy, a, b, hx, hy, view)
