"""Home task evaluation: walk across the room around obstacles to the counter, lock the base there, tidy it.

    python scripts/home_eval.py --perception yolo --episodes 16
    python scripts/home_eval.py --task home_conveyor --perception yolo --episodes 16

Writes data/home_eval_<perception>.json: navigation (arrived, stop-point error, bumps, falls, walking time) and
the tidy-table outcome per object, with Wilson intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(args: tuple) -> dict:
    seed, perception, task = args
    from homehand.control.home_controller import HomeController
    from homehand.env.home_env import HomeConveyorEnv, HomeEnv
    from homehand.runner import run_episode
    env = HomeConveyorEnv() if task == "home_conveyor" else HomeEnv()
    ctl = HomeController(env, perception=perception)
    t0 = time.time()
    r = run_episode(env, ctl, task=task, seed=seed)
    return {"seed": seed, "outcome": r["outcome"], "objects": {str(k): v for k, v in r["object_outcomes"].items()},
            "nav": env.info.details.get("nav", {}), "n_obstacles": len(env.obstacles),
            "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    from homehand import paths
    from homehand.eval.metrics import wilson
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="home", choices=["home", "home_conveyor"])
    ap.add_argument("--perception", default="yolo", choices=["ground_truth", "oracle", "yolo"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    eps = []
    # a fresh process per episode: a worker that had run three episodes lost all four objects of the fourth
    # (twice), while the same seed alone cleared 4/4 - state leaks between episodes in one process
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for ep in pool.imap(run, [(s, a.perception, a.task) for s in range(a.episodes)]):
            nav = ep["nav"]
            print(ep["seed"], ep["outcome"], ep["objects"], f"arrived={nav.get('arrived')} err={nav.get('dock_error_m')} "
                  f"bumps={nav.get('bumps')} fell={nav.get('fell')} walk={nav.get('walk_time_s')}s", flush=True)
            eps.append(ep)
    objs = Counter(v for e in eps for v in e["objects"].values())
    n_obj = sum(objs.values())
    clean = sum(all(v == "success" for v in e["objects"].values()) for e in eps)
    navs = [e["nav"] for e in eps]
    arrived = [n for n in navs if n.get("arrived")]
    summary = {
        "config": vars(a),
        "arrived": wilson(len(arrived), len(eps)),
        "no_bump": wilson(sum(n.get("bumps", 0) == 0 for n in navs), len(eps)),
        "falls": sum(bool(n.get("fell")) for n in navs),
        "dock_error_cm": {"median": round(100 * sorted(n["dock_error_m"] for n in arrived)[len(arrived) // 2], 1),
                          "max": round(100 * max(n["dock_error_m"] for n in arrived), 1)} if arrived else None,
        "walk_time_s_mean": round(sum(n["walk_time_s"] for n in arrived) / len(arrived), 1) if arrived else None,
        "objects_in_bin": wilson(objs["success"], n_obj), "clean_episodes": wilson(clean, len(eps)),
        "object_outcomes": dict(objs), "episodes": eps,
    }
    out = paths.PROJECT_ROOT / "data" / (f"home_eval_{a.perception}.json" if a.task == "home"
                                         else f"home_conveyor_eval_{a.perception}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2, default=str))


if __name__ == "__main__":
    main()
