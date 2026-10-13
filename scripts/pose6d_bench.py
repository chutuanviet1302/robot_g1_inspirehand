"""Accuracy of the 6-D pose estimate (`homehand.perception.pose6d`) on the conveyor, against the simulator's poses.

    python scripts/pose6d_bench.py --scenes 75 --detector oracle,yolo --workers 3

Scenes: the belt stopped, 4 objects upright or lying at any heading in the pick zones (`ConveyorEnv`, speed 0).
Per object: detected or not, rest pose right or not, centroid error, ADD-S (symmetric objects), and the heading error
that matters to the grasp (boxes upright: mod 90 deg - either pair of faces; round objects upright: none; lying:
the long axis, mod 180 deg). Writes data/pose6d_bench_<detector>.json and prints a summary per object and rest pose.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def heading_error(name: str, rest: str, est: float, true: float) -> float | None:
    from homehand.model.objects import OBJECTS
    if rest == "upright":
        if not OBJECTS[name].box:
            return None
        period = np.pi / 2
    else:
        period = np.pi
    d = (est - true) % period
    return float(min(d, period - d))


def run(args: tuple) -> list[dict]:
    seeds, detector = args
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    import time
    from homehand.env.conveyor_env import BELT_Z, ConveyorEnv
    from homehand.model.objects import LYING
    from homehand.perception.camera import HeadCamera
    from homehand.perception.detector import make_detector
    from homehand.perception.pose6d import estimate_pose, lying_axis, pose_errors, quat2mat
    env = ConveyorEnv()
    cam = HeadCamera(env)
    det = make_detector(detector, env)
    cache, out = {}, []
    for seed in seeds:
        env.reset(seed=seed, speed=0.0)
        for _ in range(10):
            env.step(env.home_ctrl)
        frame = cam.capture(with_seg=det.kind == "oracle")
        dets = {d.name: d for d in det.detect(frame)}
        for name, x, y, yaw, rest in env.layout:
            p, q = env.object_pose(name)
            R_true = quat2mat(q)
            if rest == "upright":
                true_heading = float(np.arctan2(R_true[1, 0], R_true[0, 0]))
            else:
                ax = R_true @ lying_axis(name)
                true_heading = float(np.arctan2(ax[1], ax[0]))
            row = {"seed": seed, "object": name, "rest": rest, "detected": name in dets}
            if name in dets:
                t0 = time.perf_counter()
                est = estimate_pose(dets[name], frame, env.model, env.obj_info, BELT_Z, cache)
                row["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
                if est is not None:
                    e = pose_errors(cache[name], est.R, est.pos, R_true, p)
                    he = heading_error(name, rest, est.heading, true_heading) if est.rest == rest else None
                    row.update(estimated=True, rest_ok=est.rest == rest, centre_mm=round(e["centre"] * 1e3, 2),
                               add_s_mm=round(e["add_s"] * 1e3, 2),
                               heading_deg=None if he is None else round(float(np.rad2deg(he)), 2),
                               rmse_mm=round(est.rmse * 1e3, 2), inliers=round(est.inliers, 3))
                else:
                    row["estimated"] = False
            out.append(row)
    return out


def summarise(rows: list[dict]) -> dict:
    def stats(rs):
        ok = [r for r in rs if r.get("estimated")]
        c = np.array([r["centre_mm"] for r in ok]) if ok else np.zeros(1)
        a = np.array([r["add_s_mm"] for r in ok]) if ok else np.zeros(1)
        h = np.array([r["heading_deg"] for r in ok if r.get("heading_deg") is not None])
        return {"n": len(rs), "detected": sum(r["detected"] for r in rs), "estimated": len(ok),
                "rest_ok": sum(r.get("rest_ok", False) for r in ok),
                "centre_mm_median": round(float(np.median(c)), 1), "centre_mm_p90": round(float(np.percentile(c, 90)), 1),
                "add_s_mm_median": round(float(np.median(a)), 1), "add_s_mm_p90": round(float(np.percentile(a, 90)), 1),
                "heading_deg_median": round(float(np.median(h)), 1) if len(h) else None,
                "heading_deg_p90": round(float(np.percentile(h, 90)), 1) if len(h) else None,
                "ms_median": round(float(np.median([r["ms"] for r in rs if "ms" in r])), 0) if any("ms" in r for r in rs) else None}
    out = {"all": stats(rows)}
    for key in sorted({(r["object"], r["rest"]) for r in rows}):
        out[f"{key[0]} ({key[1]})"] = stats([r for r in rows if (r["object"], r["rest"]) == key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=75)
    ap.add_argument("--detector", default="oracle")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from homehand.resources import WORKER_TORCH_MB, safe_workers
    for detector in a.detector.split(","):
        w = safe_workers(a.workers, per_worker_mb=WORKER_TORCH_MB if detector == "yolo" else 1200)
        chunks = [list(range(i, a.scenes, w)) for i in range(w)]
        with mp.get_context("spawn").Pool(w) as pool:
            rows = [r for part in pool.map(run, [(c, detector) for c in chunks]) for r in part]
        summary = summarise(rows)
        print(detector, json.dumps(summary["all"]))
        for k, v in summary.items():
            if k != "all":
                print(f"   {k:22s} {v}")
        (ROOT / "data" / f"pose6d_bench_{detector}.json").write_text(
            json.dumps({"args": vars(a), "summary": summary, "objects": rows}, indent=1))


if __name__ == "__main__":
    main()
