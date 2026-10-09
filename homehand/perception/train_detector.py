"""Synthetic training data + training for the head-camera detector (YOLOv8n-seg).

Images are rendered from the head camera with automatic instance-mask labels from MuJoCo's segmentation
buffer. Domain randomisation: object subset / pose, lighting intensity and direction, a few mm / degrees of
camera pose jitter, and random arm poses so the hands occlude objects as they do during the task.

    homehand perception gen-data --n 1500
    homehand perception train --epochs 40
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from homehand import paths
from homehand.model.objects import CLASS_NAMES

YOLO_DIR = paths.DATA_DIR / "perception" / "yolo"


def _mask_polygon(mask: np.ndarray) -> np.ndarray | None:
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 80:
        return None
    c = cv2.approxPolyDP(c, 1.0, True)
    return c.reshape(-1, 2) if len(c) >= 3 else None


def gen_data(n: int = 1500, seed: int = 0, val_frac: float = 0.15) -> Path:
    import os
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
    from PIL import Image

    from homehand.env.kitchen_env import KitchenEnv
    from homehand.perception.camera import HeadCamera
    from homehand.perception.detector import OracleDetector

    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True)
    env = KitchenEnv()
    cam = HeadCamera(env)
    oracle = OracleDetector(env)
    m = env.model
    cam_pos0 = m.cam_pos[cam.cam_id].copy()
    cam_quat0 = m.cam_quat[cam.cam_id].copy()
    light_dif0 = m.light_diffuse.copy()
    light_dir0 = m.light_dir.copy()
    rng = np.random.default_rng(seed)
    counts = {c: 0 for c in CLASS_NAMES}
    for i in range(n):
        k = int(rng.integers(1, len(CLASS_NAMES) + 1))
        names = list(rng.choice(CLASS_NAMES, size=k, replace=False))
        env.reset("tidy_table", seed=seed * 100_000 + i, objects=names)
        # random arm poses: the hands often hide part of the counter during the task
        if rng.random() < 0.5:
            env.data.qpos[env.arm_qadr] += rng.normal(0, 0.25, 14)
            mujoco.mj_forward(m, env.data)
        # lighting + camera jitter
        m.light_diffuse[:] = light_dif0 * rng.uniform(0.6, 1.4)
        m.light_dir[:] = light_dir0 + rng.normal(0, 0.15, light_dir0.shape)
        m.cam_pos[cam.cam_id] = cam_pos0 + rng.normal(0, 0.008, 3)
        dq = np.zeros(4)
        mujoco.mju_euler2Quat(dq, rng.normal(0, np.deg2rad(1.5), 3), "xyz")
        mujoco.mju_mulQuat(m.cam_quat[cam.cam_id], cam_quat0, dq)
        mujoco.mj_forward(m, env.data)
        frame = cam.capture(with_seg=True)
        split = "val" if rng.random() < val_frac else "train"
        stem = f"img{i:05d}"
        Image.fromarray(frame.rgb).save(YOLO_DIR / "images" / split / f"{stem}.jpg", quality=92)
        h, w = frame.rgb.shape[:2]
        lines = []
        for det in oracle.detect(frame):
            poly = _mask_polygon(det.mask)
            if poly is None:
                continue
            cls = CLASS_NAMES.index(det.name)
            counts[det.name] += 1
            coords = " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly)
            lines.append(f"{cls} {coords}")
        (YOLO_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        if (i + 1) % 200 == 0:
            print(f"[perception] rendered {i + 1}/{n}")
    m.cam_pos[cam.cam_id], m.cam_quat[cam.cam_id] = cam_pos0, cam_quat0
    m.light_diffuse[:], m.light_dir[:] = light_dif0, light_dir0
    env.close()
    yaml = YOLO_DIR / "data.yaml"
    yaml.write_text(f"path: {YOLO_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
                    + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES)))
    (YOLO_DIR / "stats.json").write_text(json.dumps({"images": n, "instances": counts}, indent=2))
    print(f"[perception] dataset at {YOLO_DIR} ({counts})")
    return yaml


def train(epochs: int = 40, imgsz: int = 512, batch: int = 8, base: str = "yolov8n-seg.pt") -> Path:
    from ultralytics import YOLO

    from homehand.perception.detector import YOLO_WEIGHTS

    yaml = YOLO_DIR / "data.yaml"
    if not yaml.exists():
        raise SystemExit("no dataset - run `homehand perception gen-data` first")
    model = YOLO(base)
    project = paths.DATA_DIR / "perception" / "runs"
    model.train(data=str(yaml), epochs=epochs, imgsz=imgsz, batch=batch, workers=2, project=str(project),
                name="yolo_seg", exist_ok=True, plots=False, verbose=False, cache=False, amp=False)  # fp16 is unreliable on GTX 16xx cards
    best = project / "yolo_seg" / "weights" / "best.pt"
    YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, YOLO_WEIGHTS)
    metrics = YOLO(str(YOLO_WEIGHTS)).val(data=str(yaml), imgsz=imgsz, batch=batch, workers=2, verbose=False,
                                          plots=False)
    summary = {"box_map50": float(metrics.box.map50), "box_map50_95": float(metrics.box.map),
               "mask_map50": float(metrics.seg.map50), "mask_map50_95": float(metrics.seg.map),
               "epochs": epochs, "imgsz": imgsz}
    (YOLO_WEIGHTS.parent / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[perception] saved {YOLO_WEIGHTS}: {summary}")
    return YOLO_WEIGHTS


def localization_benchmark(detector: str = "yolo", n: int = 60) -> dict:
    """Detection recall and 3D localisation error of detector + depth vs. ground truth."""
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.model.objects import OBJECTS, grasp_geometry
    from homehand.perception.camera import HeadCamera
    from homehand.perception.detector import make_detector
    from homehand.perception.localize import localize

    env = KitchenEnv()
    cam = HeadCamera(env)
    det = make_detector(detector, env)
    errs, yaw_errs, found, total, ms = [], [], 0, 0, []
    rest_ok = rest_n = 0
    import time
    for s in range(n):
        obs = env.reset("tidy_table", seed=900_000 + s)
        frame = cam.capture(with_seg=det.kind == "oracle")
        t0 = time.perf_counter()
        dets = det.detect(frame)
        ms.append((time.perf_counter() - t0) * 1e3)
        total += len(obs["objects"])
        for d in dets:
            if d.name not in obs["objects"]:
                continue
            e = localize(d, frame)
            if e is None:
                continue
            found += 1
            gt = obs["objects"][d.name]
            rest_n += 1
            rest_ok += e.rest == gt["rest"]
            ref = gt["centre"] if gt["rest"] == "lying" else gt["pos"]
            errs.append(float(np.linalg.norm(e.pos[:2] - ref[:2])))
            if gt["rest"] == "lying" and e.rest == "lying":
                yaw_errs.append(abs((e.yaw - gt["yaw"] + np.pi / 2) % np.pi - np.pi / 2))
            elif OBJECTS[d.name].box and gt["rest"] == "upright":
                a = grasp_geometry(OBJECTS[d.name], e.yaw)[0] - grasp_geometry(OBJECTS[d.name], gt["yaw"])[0]
                yaw_errs.append(abs((a + np.pi / 2) % np.pi - np.pi / 2))
    env.close()
    errs = np.array(errs)
    return {"detector": det.kind, "recall": round(found / max(total, 1), 4),
            "pos_err_mean_mm": round(1e3 * errs.mean(), 1), "pos_err_p90_mm": round(1e3 * np.percentile(errs, 90), 1),
            "yaw_err_mean_deg": round(float(np.degrees(np.mean(yaw_errs))), 1) if yaw_errs else 0.0,
            "rest_pose_accuracy": round(rest_ok / max(rest_n, 1), 4),
            "detect_ms": round(float(np.mean(ms)), 1)}
