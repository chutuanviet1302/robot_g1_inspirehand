"""Conveyor task evaluation: episodes with N objects on the running belt, per-object outcomes with Wilson intervals.

    python scripts/conveyor_eval.py --speed 0.03 --pose oracle --objects 4 --episodes 8
    python scripts/conveyor_eval.py --speed 0.03 --pose ground_truth --objects 2 --episodes 10

Writes data/conveyor_eval_<pose>_<n>obj_<speed>.json. Episodes run in parallel worker processes (each needs ~0.8 GB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(args: tuple) -> dict:
    seed, speed, pose, n, gap = args
    from homehand.control.conveyor_planner import ConveyorPlanner
    from homehand.env.conveyor_env import ConveyorEnv
    from homehand.runner import run_episode
    env = ConveyorEnv()
    planner = ConveyorPlanner(env, perception=pose)
    reset = env.reset
    env.reset = lambda task="conveyor", seed=0, objects=None, randomization=None, layout=None: reset(
        task, seed=seed, speed=speed, n_objects=n, gap=gap)
    t0 = time.time()
    r = run_episode(env, planner, task="conveyor", seed=seed)
    stops = [e.get("reason") for e in r["events"] if e.get("event") == "protective_stop"]
    return {"seed": seed, "outcomes": {str(k): v for k, v in r["object_outcomes"].items()},
            "protective_stop": stops[0] if stops else None, "steps": r["steps"],
            "wall_s": round(time.time() - t0, 1), "perception": dict(planner.perception.stats)}


def main() -> None:
    from homehand import paths
    from homehand.eval.metrics import wilson
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speed", type=float, default=0.03, help="belt speed, m/s")
    ap.add_argument("--pose", default="oracle", choices=["ground_truth", "oracle", "yolo"],
                    help="ground-truth poses, or 6-D poses fitted on oracle / YOLO masks")
    ap.add_argument("--objects", type=int, default=4, help="objects per episode")
    ap.add_argument("--gap", default="0.40,0.55", help="min,max distance between objects on the belt (m)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    gap = tuple(float(g) for g in a.gap.split(","))
    jobs = [(s, a.speed, a.pose, a.objects, gap) for s in range(a.episodes)]
    episodes = []
    with get_context("spawn").Pool(a.workers) as pool:
        for ep in pool.imap(run, jobs):
            print(ep["seed"], ep["outcomes"], ep["protective_stop"] or "", f"{ep['wall_s']:.0f}s", flush=True)
            episodes.append(ep)
    outcomes = Counter(v for ep in episodes for v in ep["outcomes"].values())
    per_object: dict = {}
    for ep in episodes:
        for name, v in ep["outcomes"].items():
            k, n = per_object.get(name, (0, 0))
            per_object[name] = (k + (v == "success"), n + 1)
    n_obj = sum(outcomes.values())
    clean = sum(all(v == "success" for v in ep["outcomes"].values()) for ep in episodes)
    summary = {"config": vars(a), "objects_in_bin": wilson(outcomes["success"], n_obj),
               "clean_episodes": wilson(clean, len(episodes)), "outcomes": dict(outcomes),
               "per_object": {k: f"{s}/{n}" for k, (s, n) in per_object.items()},
               "protective_stops": sum(ep["protective_stop"] is not None for ep in episodes), "episodes": episodes}
    out = paths.PROJECT_ROOT / "data" / f"conveyor_eval_{a.pose}_{a.objects}obj_{a.speed:g}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
