"""Evaluate a fine-tuned SmolVLA checkpoint in the kitchen (LeRobot environment; one process, model on the GPU).

    .venv-vla\\Scripts\\python scripts/eval_smolvla.py --model models/smolvla_homehand -n 20
    .venv-vla\\Scripts\\python scripts/eval_smolvla.py --model models/smolvla_homehand -n 10 --held-out-language

Results go to the same SQLite store as `homehand eval` (they appear in the web UI and in RESULTS.md).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--perception", default="oracle")
    ap.add_argument("--randomization", default="nominal")
    ap.add_argument("--held-out-language", action="store_true", help="instructions phrased as never in training")
    a = ap.parse_args()

    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.eval import store
    from homehand.eval.metrics import summarize
    from homehand.policy.smolvla_skill import SmolVLARunner
    from homehand.runner import run_episode

    runner = SmolVLARunner(a.model, held_out_language=a.held_out_language)
    env = KitchenEnv()
    planner = TidyPlanner(env, perception=a.perception, policy=runner)
    label = "smolvla" + (" (held-out phrasing)" if a.held_out_language else "")
    run_id = store.create_run(f"{label} | {a.perception} | {a.randomization}", "tidy_table", runner.name,
                              a.perception, a.randomization, a.n, a.seed0)
    episodes, t0 = [], time.time()
    try:
        for i in range(a.n):
            ep = run_episode(env, planner, seed=a.seed0 + i, randomization=a.randomization)
            episodes.append(ep)
            store.add_episode(run_id, ep, None)
            print(f"seed {a.seed0 + i}: {ep['outcome']} {ep['n_cleared']}/{ep['n_objects']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        s = summarize(episodes)
        store.finish_run(run_id, s, time.time() - t0)
    except BaseException:
        store.finish_run(run_id, summarize(episodes) if episodes else {}, time.time() - t0, status="failed")
        raise
    print({"run": run_id, "task_success": s["task_success"], "object_clear_rate": s["object_clear_rate"],
           "episode_outcomes": s["episode_outcomes"], "safety": s["safety"]})


if __name__ == "__main__":
    main()
