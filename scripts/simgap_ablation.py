"""One sim-gap factor at a time, at the extreme of the `high` preset (expert, ground-truth poses).

    python scripts/simgap_ablation.py --seeds 8 --out data/simgap_ablation.json

The random presets mix every factor at once, so with few episodes their success rates are noisy; fixing one
factor at its worst value shows which assumption the controller actually depends on.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FACTORS = {
    "nominal": {},
    "friction x0.5": {"friction": (0.5, 0.5)},
    "friction x1.5": {"friction": (1.5, 1.5)},
    "mass x0.5": {"mass": (0.5, 0.5)},
    "mass x2.0": {"mass": (2.0, 2.0)},
    "latency 3 steps (120 ms)": {"latency": (3, 3)},
    "finger stiffness x0.6": {"finger_gain": (0.6, 0.6)},
    "pose noise 12 mm": {"obs_noise": 0.012},
}


def run(args) -> dict:
    name, seed = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.runner import run_episode
    env = KitchenEnv()
    r = run_episode(env, TidyPlanner(env, perception="ground_truth"), seed=seed, randomization=FACTORS[name])
    return {"factor": name, "seed": seed, "outcome": r["outcome"], "cleared": r["n_cleared"], "n": r["n_objects"],
            "failures": {str(k): v for k, v in r["object_outcomes"].items() if v != "success"},
            "safety_stop": bool(r["safety"].get("protective_stop"))}


def main() -> None:
    from homehand.resources import safe_workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    jobs = [(f, s) for f in FACTORS for s in range(a.seeds)]
    with mp.get_context("spawn").Pool(safe_workers(a.workers), maxtasksperchild=4) as pool:
        res = pool.map(run, jobs)
    table = {}
    for f in FACTORS:
        rs = [r for r in res if r["factor"] == f]
        modes: dict = {}
        for r in rs:
            for v in r["failures"].values():
                modes[v] = modes.get(v, 0) + 1
        table[f] = {"episodes": len(rs), "clean": sum(r["outcome"] == "success" for r in rs),
                    "objects": f"{sum(r['cleared'] for r in rs)}/{sum(r['n'] for r in rs)}",
                    "safety_stops": sum(r["safety_stop"] for r in rs), "failures": modes}
        print(f"{f:26s} clean {table[f]['clean']}/{len(rs)}  objects {table[f]['objects']}  {modes}")
    if a.out:
        Path(a.out).write_text(json.dumps({"table": table, "episodes": res}, indent=1))


if __name__ == "__main__":
    main()
