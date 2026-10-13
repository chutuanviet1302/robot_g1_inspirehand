"""Perception for the conveyor: detections -> 6-D poses -> tracks, at a fixed rate.

Rules learnt the hard way: a pose fitted to an object only partly in view (entering at the image edge) can be wrong
outright (an upright box taken for a lying one), and refining from a wrong pose never recovers. So
- an object whose mask touches the image border is not used;
- a track is refined from its prediction (fast) but fitted again from scratch when the refinement matches fewer than
  MIN_INLIERS of the observed points;
- a fit that still matches poorly is not used.

`mode`: "ground_truth" (the simulator's poses, through the same interface), "oracle" (segmentation masks) or
"yolo" (the trained detector).
"""

from __future__ import annotations

import numpy as np

from homehand.perception.pose6d import Pose6D, _result, estimate_pose, model_points, quat2mat, refine_pose
from homehand.perception.tracker import MIN_INLIERS, Tracker

BORDER = 4          # px: masks closer than this to the image edge are objects partly out of view


class ConveyorPerception:
    def __init__(self, env, mode: str = "oracle", support_z: float | None = None):
        from homehand.env.conveyor_env import BELT_Z
        self.env, self.mode = env, mode
        self.support_z = BELT_Z if support_z is None else support_z
        self.cache: dict = {}
        self.camera = self.detector = None
        if mode != "ground_truth":
            from homehand.perception.camera import HeadCamera
            from homehand.perception.detector import make_detector
            self.camera = HeadCamera(env)
            self.detector = make_detector(mode, env)
        self.reset()

    def reset(self, prior_velocity=(0.0, 0.0)) -> None:
        self.tracker = Tracker(prior_velocity)
        self.stats = {"frames": 0, "full_fits": 0, "refits": 0, "skipped_border": 0, "rejected": 0}

    @property
    def tracks(self):
        return self.tracker.tracks

    def _gt_pose(self, name: str) -> Pose6D:
        if name not in self.cache:
            self.cache[name] = model_points(self.env.model, name)
        p, q = self.env.object_pose(name)
        return _result(name, quat2mat(q), p, 0.0, 1.0, len(self.cache[name]), self.cache[name])

    def update(self, t: float) -> None:
        """One perception cycle at time t (seconds)."""
        self.stats["frames"] += 1
        if self.mode == "ground_truth":
            obs = self.env.observe()["objects"]
            self.tracker.update([self._gt_pose(n) for n, o in obs.items() if o["on_table"]], t)
            return
        frame = self.camera.capture(with_seg=self.detector.kind == "oracle")
        h, w = frame.depth.shape
        fits = []
        for d in self.detector.detect(frame):
            x1, y1, x2, y2 = d.bbox
            if x1 < BORDER or y1 < BORDER or x2 > w - BORDER or y2 > h - BORDER:
                self.stats["skipped_border"] += 1
                continue
            fit = None
            track = self.tracker.tracks.get(d.name)
            if track is not None:
                fit = refine_pose(track.predict(t), d, frame, self.env.model, self.support_z, self.cache)
            if fit is None or fit.inliers < MIN_INLIERS:
                if track is not None:
                    self.stats["refits"] += 1
                    self.tracker.drop(d.name)            # its pose is no longer trusted: start it again
                fit = estimate_pose(d, frame, self.env.model, self.env.obj_info, self.support_z, self.cache)
                self.stats["full_fits"] += 1
            if fit is None or fit.inliers < MIN_INLIERS:
                self.stats["rejected"] += 1
                continue
            fits.append(fit)
        self.tracker.update(fits, t)
