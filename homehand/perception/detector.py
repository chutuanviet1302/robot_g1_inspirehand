"""Object detectors for the head camera.

* `YoloDetector`  - YOLOv8n-seg trained on synthetic head-camera images (`homehand perception train`).
* `OracleDetector` - reads the simulator's segmentation buffer. Used to auto-label the training set and as an
                     upper-bound baseline ("perfect perception") in the evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from homehand import paths
from homehand.model.objects import CLASS_NAMES
from homehand.perception.camera import Frame
from homehand.perception.localize import Detection

YOLO_WEIGHTS = paths.MODELS_DIR / "detector" / "yolo_seg.pt"
MIN_PIXELS = 150


class OracleDetector:
    kind = "oracle"

    def __init__(self, env):
        self.env = env
        self.geom_obj = {}
        for name, geoms in env.obj_geoms.items():
            for g in geoms:
                self.geom_obj[g] = name

    def detect(self, frame: Frame) -> list[Detection]:
        assert frame.seg is not None, "OracleDetector needs a frame captured with_seg=True"
        out = []
        ids = np.full(frame.seg.shape, -1, dtype=np.int32)
        for g, name in self.geom_obj.items():
            ids[frame.seg == g] = CLASS_NAMES.index(name)
        for c, name in enumerate(CLASS_NAMES):
            mask = ids == c
            n = int(mask.sum())
            if n < MIN_PIXELS:
                continue
            vs, us = np.nonzero(mask)
            out.append(Detection(name, 1.0, (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1), mask))
        return out


class YoloDetector:
    kind = "yolo"

    def __init__(self, weights: Path | str = YOLO_WEIGHTS, conf: float = 0.4, device: str | None = None):
        from ultralytics import YOLO  # optional dependency
        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device

    def detect(self, frame: Frame) -> list[Detection]:
        h, w = frame.rgb.shape[:2]
        res = self.model.predict(frame.rgb[..., ::-1], conf=self.conf, verbose=False, device=self.device,
                                 retina_masks=True)[0]  # ultralytics expects BGR numpy images
        out = []
        if res.boxes is None:
            return out
        masks = res.masks.data.cpu().numpy() if res.masks is not None else None
        best: dict[str, Detection] = {}
        for i, (box, cls, score) in enumerate(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                                  res.boxes.conf.cpu().numpy())):
            name = CLASS_NAMES[int(cls)]
            mask = None
            if masks is not None:
                mk = masks[i]
                mask = (mk > 0.5) if mk.shape == (h, w) else _resize_mask(mk, h, w)
            det = Detection(name, float(score), tuple(int(v) for v in box), mask)
            if name not in best or det.score > best[name].score:  # one instance per class in this task
                best[name] = det
        return list(best.values())


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = (np.arange(h) * mask.shape[0] / h).astype(int)
    xs = (np.arange(w) * mask.shape[1] / w).astype(int)
    return mask[ys][:, xs] > 0.5


def make_detector(kind: str, env):
    """kind: 'yolo' | 'oracle' | 'auto' (yolo when trained weights exist, oracle otherwise)."""
    if kind == "auto":
        kind = "yolo" if YOLO_WEIGHTS.exists() else "oracle"
    if kind == "yolo":
        try:
            return YoloDetector()
        except Exception as e:  # missing ultralytics / weights
            raise RuntimeError(f"YOLO detector unavailable ({e}); train it with `homehand perception train`") from e
    return OracleDetector(env)
