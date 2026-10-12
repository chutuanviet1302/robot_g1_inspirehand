"""Evaluate the bimanual box lift (`env.bimanual_env`, `control.bimanual`) over seeds and randomisation presets.

    python scripts/bimanual_eval.py --seeds 0-49 --randomization nominal,medium --workers 4

Writes data/bimanual_eval_<tag>.json: per preset the success rate (Wilson 95 % CI), outcome counts, lift height,
goal error, tilt and the largest contact force; and every episode.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_W: dict = {}


def run_one(args: tuple[int, str]) -> dict:
    seed, randomization = args
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    from homehand.control.bimanual import BimanualExpert
    from homehand.env.bimanual_env import BimanualEnv
    from homehand.runner import run_episode
    if "env" not in _W:
        _W["env"] = BimanualEnv()
    env = _W["env"]
    r = run_episode(env, BimanualExpert(env), task="bimanual_lift", seed=seed, randomization=randomization)
    d = env.info.details
    return {"seed": seed, "randomization": randomization, "outcome": r["outcome"], "max_lift": d["max_lift"],
            "held_steps": d["held_steps"], "goal_error": d["goal_error"], "tilt_deg": d["tilt_deg"],
            "max_contact_force": float(d["safety"]["max_contact_force"]), "sampled": d["sampled"]}


def main() -> None:
    from homehand.eval.metrics import wilson
    from homehand.resources import safe_workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--randomization", default="nominal,medium")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    jobs = [(s, r) for r in a.randomization.split(",") for s in seeds]
    with mp.get_context("spawn").Pool(safe_workers(a.workers)) as pool:
        res = sorted(pool.map(run_one, jobs), key=lambda e: (e["randomization"], e["seed"]))
    summary = {}
    for r in a.randomization.split(","):
        eps = [e for e in res if e["randomization"] == r]
        ok = [e for e in eps if e["outcome"] == "success"]
        summary[r] = {"success": wilson(len(ok), len(eps)), "outcomes": dict(Counter(e["outcome"] for e in eps)),
                      "mean_max_lift": round(sum(e["max_lift"] for e in eps) / len(eps), 4),
                      "mean_goal_error_ok": round(sum(e["goal_error"] for e in ok) / max(len(ok), 1), 4),
                      "max_tilt_ok": max((e["tilt_deg"] for e in ok), default=None)}
    print(json.dumps(summary, indent=1))
    out = ROOT / "data" / f"bimanual_eval_{a.tag}.json"
    out.write_text(json.dumps({"args": vars(a), "summary": summary, "episodes": res}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
