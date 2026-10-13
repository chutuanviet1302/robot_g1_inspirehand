"""6-D object pose from one RGB-D view: mask + depth -> point cloud -> ICP against the object's mesh.

1. The object's pixels (YOLO or oracle mask) are back-projected to world points (`camera.deproject`); points on
   the support surface are dropped.
2. Initial guesses: the footprint fit (`localize`: rest pose, position, heading) turned into a body pose with the
   same function that places objects in the scene (`objects.body_pose_for`), then every turn of the heading by 90 deg
   and the upside-down flip of each, so the symmetric-looking faces of a box or can cannot trap the fit.
3. Point-to-point ICP of the observed points against the mesh points that face the camera (an observed cloud only
   shows that side: matching it to the far side would pull the fit through the object), with a shrinking
   correspondence gate; the candidate with the lowest residual wins.

Output: the body pose (position, rotation) in the world, the residual, the share of observed points within 5 mm
of the fitted model, and the rest pose (`upright` / `lying`, from where the object's own up axis points) and
heading the grasp library uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from homehand.model.objects import body_pose_for, lying_dims, rest_options
from homehand.perception.camera import Frame, deproject
from homehand.perception.localize import Detection, localize

MODEL_POINTS = 1500
ICP_ITERS = 20
GATES = (0.03, 0.02, 0.012, 0.008)       # m, correspondence distance limit, tightened over the iterations
INLIER = 0.005


def quat2mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def rot(axis, angle) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    return quat2mat(q)


def model_points(model: mujoco.MjModel, name: str, n: int = MODEL_POINTS, seed: int = 0) -> np.ndarray:
    """Points on the object's visual mesh, in its body frame."""
    mid = model.mesh(f"{name}_mesh").id
    g = next(g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
             and model.geom_dataid[g] == mid)
    a, k = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[a:a + k]
    v = v[np.random.default_rng(seed).choice(len(v), min(n, len(v)), replace=False)]
    return model.geom_pos[g] + v @ quat2mat(model.geom_quat[g]).T


@dataclass
class Pose6D:
    name: str
    pos: np.ndarray            # body origin (world)
    R: np.ndarray              # body rotation (world)
    rmse: float
    inliers: float             # share of observed points within INLIER of the fitted model
    n_points: int
    rest: str = "upright"
    heading: float = 0.0       # yaw of the object's long / x axis on the support, as the grasp library expects
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))   # model centroid (world)

    @property
    def quat(self) -> np.ndarray:
        return mat2quat(self.R)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |R P + t - Q| (rows are points)."""
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def icp(obs: np.ndarray, model: np.ndarray, R: np.ndarray, t: np.ndarray, cam: np.ndarray,
        iters: int = ICP_ITERS) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit the body pose (R, t) so that the camera-facing model points match the observed points."""
    centroid = model.mean(0)
    for i in range(iters):
        world = model @ R.T + t
        out = (world - (R @ centroid + t))                      # outward direction (convex objects)
        visible = np.einsum("ij,ij->i", out, cam - world) > 0
        vis = world[visible] if visible.sum() > 20 else world
        dist, idx = cKDTree(vis).query(obs)
        gate = GATES[min(i * len(GATES) // iters, len(GATES) - 1)]
        keep = dist < gate
        if keep.sum() < 10:
            break
        # move the model towards the observation: solve for the update of the matched model points
        dR, dt = _kabsch(vis[idx[keep]], obs[keep])
        R, t = dR @ R, dR @ t + dt
    world = model @ R.T + t
    dist, _ = cKDTree(world).query(obs)
    return R, t, float(np.sqrt(np.mean(np.minimum(dist, 0.03) ** 2))), float(np.mean(dist < INLIER))


def observed_points(det: Detection, frame: Frame, support_z: float, max_points: int = 1500) -> np.ndarray:
    x1, y1, x2, y2 = det.bbox
    if det.mask is not None:
        vs, us = np.nonzero(det.mask)
    else:
        vv, uu = np.mgrid[y1:y2, x1:x2]
        vs, us = vv.ravel(), uu.ravel()
    obs = deproject(frame, us, vs)
    obs = obs[obs[:, 2] > support_z + 0.004]
    if len(obs) > max_points:
        obs = obs[np.random.default_rng(0).choice(len(obs), max_points, replace=False)]
    return obs


def _result(name: str, R, t, rmse, inl, n, pts) -> Pose6D:
    up = R.T @ np.array([0.0, 0.0, 1.0])           # world up in the object's frame
    opts = rest_options(name)
    rest = opts[0] if len(opts) == 1 else ("upright" if abs(up[2]) > 0.7 else "lying")
    if rest == "upright":
        heading = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        axis = R @ lying_axis(name)                 # the long axis in the world
        heading = float(np.arctan2(axis[1], axis[0]))
    return Pose6D(name, t, R, rmse, inl, n, rest, heading, R @ pts.mean(0) + t)


def estimate_pose(det: Detection, frame: Frame, model: mujoco.MjModel, obj_info: dict, support_z: float,
                  cache: dict | None = None, min_points: int = 60) -> Pose6D | None:
    """Full search (a new object): 16 initial guesses, a short ICP from each, the best two refined."""
    est = localize(det, frame, min_points=min_points)
    if est is None:
        return None
    obs = observed_points(det, frame, support_z)
    if len(obs) < min_points:
        return None
    pts = _model(cache, model, det.name)
    height = float(obj_info[det.name]["height"])
    cands = []
    for rest in [est.rest] + [r for r in rest_options(det.name) if r != est.rest]:
        for k in range(4):
            p0, q0 = body_pose_for(det.name, rest, est.pos[0], est.pos[1], est.yaw + k * np.pi / 2, height, support_z)
            R0 = quat2mat(q0)
            for flip in (False, True):
                Ri = R0 @ rot([1, 0, 0], np.pi) if flip else R0
                ti = np.asarray(p0, float) + (R0 - Ri) @ pts.mean(0)      # flip about the model's centroid
                R, t, rmse, inl = icp(obs, pts, Ri, ti, frame.cam_pos, iters=6)
                cands.append((rmse - 0.002 * inl - (0.0005 if rest == est.rest else 0.0), R, t))
    cands.sort(key=lambda c: c[0])
    best = None
    for _, R0, t0 in cands[:2]:
        R, t, rmse, inl = icp(obs, pts, R0, t0, frame.cam_pos)
        if best is None or rmse - 0.002 * inl < best[0]:
            best = (rmse - 0.002 * inl, R, t, rmse, inl)
    _, R, t, rmse, inl = best
    return _result(det.name, R, t, rmse, inl, len(obs), pts)


def refine_pose(prior: Pose6D, det: Detection, frame: Frame, model: mujoco.MjModel, support_z: float,
                cache: dict | None = None, min_points: int = 60, iters: int = 10) -> Pose6D | None:
    """Tracking: ICP from a predicted pose only (the object moved a little since the last fit)."""
    obs = observed_points(det, frame, support_z)
    if len(obs) < min_points:
        return None
    pts = _model(cache, model, det.name)
    R, t, rmse, inl = icp(obs, pts, prior.R, prior.pos, frame.cam_pos, iters=iters)
    return _result(det.name, R, t, rmse, inl, len(obs), pts)


def _model(cache: dict | None, model: mujoco.MjModel, name: str) -> np.ndarray:
    if cache is None:
        return model_points(model, name)
    if name not in cache:
        cache[name] = model_points(model, name)
    return cache[name]


def lying_axis(name: str) -> np.ndarray:
    """The object's long axis in its body frame (the axis that lies horizontal when it is tipped over)."""
    from homehand.model.objects import LYING
    return np.asarray(LYING[name]["axis"], float)


# ----------------------------------------------------------------------------------------------- errors
def pose_errors(model_pts: np.ndarray, est_R, est_t, true_R, true_t) -> dict:
    """Centroid error, ADD-S (mean distance of each fitted model point to the closest true model point: the
    standard measure for symmetric objects) and ADD (mean point-to-point distance), all in metres."""
    a = model_pts @ np.asarray(est_R).T + est_t
    b = model_pts @ np.asarray(true_R).T + true_t
    return {"centre": float(np.linalg.norm(a.mean(0) - b.mean(0))),
            "add_s": float(np.mean(cKDTree(b).query(a)[0])),
            "add": float(np.mean(np.linalg.norm(a - b, axis=1)))}
