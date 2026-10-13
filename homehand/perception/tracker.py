"""Tracking objects on the conveyor: a constant-velocity Kalman filter on each object's centroid + its last 6-D pose.

One track per object class (each class appears once in this task, as the detector assumes). The filter state is
(x, y, vx, vy) of the model centroid; the pose's rotation is kept from the last fit (objects ride the belt without
turning). `predict(t)` gives the pose expected at time t: used as the starting point of the next ICP refinement
and by the planner to aim where the object will be when the hand closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from homehand.perception.pose6d import Pose6D

POS_STD = 0.004          # m, measurement noise of the fitted centroid
ACC_STD = 0.01           # m/s^2, process noise (the belt runs at a constant speed)
MIN_INLIERS = 0.6        # fits matching fewer observed points (an object half in view, occluded) are not used


@dataclass
class Track:
    name: str
    pose: Pose6D                         # last fit
    t: float                             # time of the state (s)
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    P: np.ndarray = field(default_factory=lambda: np.diag([POS_STD ** 2] * 2 + [0.05 ** 2] * 2))
    n_updates: int = 0

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]

    def _propagate(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        return F @ self.x, F @ self.P @ F.T + G @ G.T * ACC_STD ** 2

    def update(self, pose: Pose6D, t: float) -> None:
        x, P = self._propagate(t - self.t)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ P @ H.T + np.eye(2) * POS_STD ** 2
        K = P @ H.T @ np.linalg.inv(S)
        self.x = x + K @ (pose.centre[:2] - H @ x)
        self.P = (np.eye(4) - K @ H) @ P
        self.pose, self.t = pose, t
        self.n_updates += 1

    def predict(self, t: float) -> Pose6D:
        """The last pose moved to where the filter expects the object at time t."""
        x, _ = self._propagate(t - self.t)
        shift = np.array([x[0] - self.pose.centre[0], x[1] - self.pose.centre[1], 0.0])
        p = self.pose
        return Pose6D(p.name, p.pos + shift, p.R, p.rmse, p.inliers, p.n_points, p.rest, p.heading, p.centre + shift)


class Tracker:
    def __init__(self, prior_velocity=(0.0, 0.0)):
        self.tracks: dict[str, Track] = {}
        self.prior_velocity = np.asarray(prior_velocity, float)     # e.g. the belt's nominal speed

    def update(self, poses: list[Pose6D | None], t: float) -> None:
        """New fits at time t (None: a detection the pose fit could not use, e.g. an object half out of view)."""
        for pose in poses:
            if pose is None or pose.inliers < MIN_INLIERS:
                continue
            tr = self.tracks.get(pose.name)
            if tr is None:
                tr = Track(pose.name, pose, t)
                tr.x = np.array([*pose.centre[:2], *self.prior_velocity])
                tr.n_updates = 1
                self.tracks[pose.name] = tr
            else:
                tr.update(pose, t)

    def drop(self, name: str) -> None:
        self.tracks.pop(name, None)
