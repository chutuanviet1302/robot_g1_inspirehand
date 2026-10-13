"""Pick success of the expert alone (zero residual) or of a trained residual on the RL episode distribution
(`homehand.rl.residual_env.ResidualPickEnv`), so the learning curve has its reference level.

    python scripts/rl_baseline.py --episodes 300 --randomization medium [--policy models/residual_ppo]
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


def run(args: tuple) -> list[dict]:
    worker, n, randomization, policy, seed0 = args
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    import numpy as np
    from homehand.rl.residual import ACT_DIM, ResidualPolicy
    from homehand.rl.residual_env import ResidualPickEnv
    pol = ResidualPolicy(policy) if policy else None
    env = ResidualPickEnv(randomization=randomization, seed=seed0 + worker)
    out = []
    for _ in range(n):
        obs, _ = env.reset()
        done = trunc = False
        ret = 0.0
        while not (done or trunc):
            a = pol.act(obs) if pol else np.zeros(ACT_DIM)
            obs, r, done, trunc, info = env.step(a)
            ret += r
        out.append({"object": info["object"], "carried": bool(info["carried"]), "return": ret,
                    "terms": info["terms"]})
    return out


def main() -> None:
    from homehand.eval.metrics import wilson
    from homehand.resources import safe_workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--randomization", default="medium")
    ap.add_argument("--policy", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=50_000, help="env seeds (training used 1000+rank)")
    a = ap.parse_args()
    w = safe_workers(a.workers)
    per = [a.episodes // w + (i < a.episodes % w) for i in range(w)]
    with mp.get_context("spawn").Pool(w) as pool:
        res = [e for part in pool.map(run, [(i, per[i], a.randomization, a.policy, a.seed) for i in range(w)])
               for e in part]
    k = sum(e["carried"] for e in res)
    out = {"policy": a.policy or "expert", "randomization": a.randomization, "carried": wilson(k, len(res)),
           "mean_return": round(sum(e["return"] for e in res) / len(res), 3),
           "fails_by_object": dict(Counter(e["object"] for e in res if not e["carried"]))}
    print(json.dumps(out, indent=1))
    tag = Path(a.policy).name if a.policy else "expert"
    (ROOT / "data" / f"rl_baseline_{tag}_{a.randomization}.json").write_text(json.dumps({"summary": out, "episodes": res}, indent=1))


if __name__ == "__main__":
    main()
