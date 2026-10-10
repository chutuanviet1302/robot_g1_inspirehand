"""Motion quality of the expert on full tidy episodes: outcome, joint smoothness and interpenetration.

    python scripts/motion_quality.py --seeds 0-23 --workers 4

Per episode: outcome, objects cleared, safety stop, RMS / peak arm joint acceleration and RMS jerk (from the
measured joint positions at 25 Hz), and the deepest interpenetration between any robot geom (visual meshes and
collision geoms) and the objects, the bin or the counter (mj_geomDistance, every second control step).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_seed(seed: int) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    import mujoco

    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.runner import run_episode

    env = KitchenEnv()
    pl = TidyPlanner(env, perception="ground_truth")
    m, d = env.model, env.data
    robot = env.safety.robot_bodies
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in robot
             and (m.geom_group[g] == 2 or m.geom_contype[g] or m.geom_conaffinity[g])]
    bin_id = m.body("bin").id
    others = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bin_id] + [m.geom("counter").id]
    for gs in env.obj_geoms.values():
        others += [g for g in gs if m.geom_contype[g] or m.geom_conaffinity[g]]
    ft = np.zeros(6)
    qs, worst = [], [0.0, ""]

    def on_step(obs, action):
        qs.append(obs["arm_q"].copy())
        if env.t % 2:
            return
        for g in geoms:
            for o in others:
                dist = mujoco.mj_geomDistance(m, d, g, o, 0.01, ft)
                if dist < worst[0]:
                    worst[0] = dist
                    worst[1] = f"{m.body(m.geom_bodyid[g]).name} vs {m.body(m.geom_bodyid[o]).name or 'counter'}"

    r = run_episode(env, pl, seed=seed, on_step=on_step)
    q = np.asarray(qs)
    acc = np.diff(q, 2, axis=0) / 0.04 ** 2
    jerk = np.diff(q, 3, axis=0) / 0.04 ** 3
    return {"seed": seed, "outcome": r["outcome"], "cleared": r["n_cleared"], "n": r["n_objects"],
            "failures": {k: v for k, v in r["object_outcomes"].items() if v != "success"},
            "safety_stop": bool(r["safety"].get("protective_stop")), "seconds": r["sim_time"],
            "rms_acc": float(np.sqrt((acc ** 2).mean())), "max_acc": float(np.abs(acc).max()),
            "rms_jerk": float(np.sqrt((jerk ** 2).mean())),
            "max_penetration_mm": round(-1000 * worst[0], 2), "penetration_pair": worst[1]}


def main() -> None:
    from homehand.resources import safe_workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-23")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    with mp.get_context("spawn").Pool(safe_workers(a.workers)) as pool:
        res = sorted(pool.map(run_seed, seeds), key=lambda r: r["seed"])
    for r in res:
        if r["outcome"] != "success":
            print(f"seed {r['seed']}: {r['outcome']} {r['cleared']}/{r['n']} {r['failures']}")
    n = len(res)
    summary = {
        "episodes": n,
        "clean": sum(r["outcome"] == "success" for r in res),
        "objects_cleared": f"{sum(r['cleared'] for r in res)}/{sum(r['n'] for r in res)}",
        "safety_stops": sum(r["safety_stop"] for r in res),
        "mean_seconds": round(float(np.mean([r["seconds"] for r in res])), 1),
        "rms_joint_acc": round(float(np.mean([r["rms_acc"] for r in res])), 2),
        "rms_joint_jerk": round(float(np.mean([r["rms_jerk"] for r in res])), 1),
        "max_penetration_mm": max(r["max_penetration_mm"] for r in res),
    }
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "episodes": res}, indent=1))


if __name__ == "__main__":
    main()
