"""Flat-box grasp: the learned policy against the expert on the same episodes (plain MuJoCo, the CPU twin of the
training env; same seeds -> same box pose, perception bias and physics for both).

    python scripts/flat_grasp_eval.py --policy models/flat_grasp/policy.npz --episodes 500
    python scripts/flat_grasp_eval.py --policy models/flat_grasp/policy.npz --episodes 200 --robot   # whole robot

Writes data/flat_grasp_eval.json: success with Wilson intervals overall and per condition (grip friction, box
mass, perception bias), and the paired outcome counts (expert only / policy only / both / neither).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

XML = ROOT / "generated/flat_grasp_train.xml"
FRAME = ROOT / "generated/flat_grasp.json"
SEED0 = 100_000                    # evaluation seeds: away from anything used while developing


def run(args: tuple) -> list[dict]:
    seeds, policy_path, robot = args
    from homehand.control.ik import ArmIK
    from homehand.rl import flat_grasp_env as E
    from homehand.rl.flat_grasp_policy import Policy
    if robot:
        from homehand.rl.flat_grasp_deploy import RobotTwin
        sim = RobotTwin()
    else:
        sim = E.FlatGraspSim(XML, FRAME)
    ik = ArmIK()
    policy = Policy(policy_path) if policy_path else None
    rows = []
    for seed in seeds:
        expert = E.run_expert(sim, seed, ik)
        row = {"seed": seed, "expert": expert, **{k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                                                   for k, v in sim.ep.items()}}
        if policy is not None:
            row["policy"] = E.run_policy(sim, policy, seed, ik)
            row["policy_lift_cm"] = round(sim.lift_cm(), 2)
        rows.append(row)
    return rows


def summary(rows: list[dict], who: list[str]) -> dict:
    from homehand.eval.metrics import wilson

    def rate(sel):
        return {w: wilson(sum(r[w] for r in sel), len(sel)) for w in who}
    bias = lambda r: float(np.hypot(*r["bias"][:2])) * 1000
    groups = {
        "all": rows,
        "slippery (grip friction < 0.9)": [r for r in rows if r["grip_friction"] < 0.9],
        "grippy (grip friction >= 1.2)": [r for r in rows if r["grip_friction"] >= 1.2],
        "heavy (mass >= 0.22 kg)": [r for r in rows if r["mass"] >= 0.22],
        "light (mass < 0.15 kg)": [r for r in rows if r["mass"] < 0.15],
        "large perception bias (>= 8 mm)": [r for r in rows if bias(r) >= 8],
    }
    out = {name: {"n": len(sel), **rate(sel)} for name, sel in groups.items() if sel}
    if "policy" in who:
        out["paired"] = {"both": sum(r["expert"] and r["policy"] for r in rows),
                         "policy_only": sum(r["policy"] and not r["expert"] for r in rows),
                         "expert_only": sum(r["expert"] and not r["policy"] for r in rows),
                         "neither": sum(not r["expert"] and not r["policy"] for r in rows)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--robot", action="store_true", help="the whole robot in the kitchen scene")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = list(range(SEED0, SEED0 + a.episodes))
    a.out = a.out or str(ROOT / f"data/flat_grasp_eval{'_robot' if a.robot else ''}.json")
    chunks = [(seeds[k::a.workers], a.policy, a.robot) for k in range(a.workers)]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        rows = sorted((r for part in pool.map(run, chunks) for r in part), key=lambda r: r["seed"])
    who = ["expert"] + (["policy"] if a.policy else [])
    s = summary(rows, who)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"policy": a.policy, "episodes": a.episodes, "robot": a.robot, "summary": s,
                                       "rows": rows},
                                      indent=1))
    for name, g in s.items():
        if name == "paired":
            print("paired:", g)
            continue
        print(f"{name:34s} n={g['n']:4d}  " + "  ".join(
            f"{w} {g[w]['p']:.2f} [{g[w]['lo']:.2f}, {g[w]['hi']:.2f}]" for w in who))
    print(f"({time.time() - t0:.0f} s) -> {a.out}")


if __name__ == "__main__":
    main()
