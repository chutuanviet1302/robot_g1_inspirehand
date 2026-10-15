"""Throughput of the flat-grasp env on MJX: policy steps per second for a batch of environments (random actions).

    python mjx_bench.py flat_grasp_train.xml [more.xml ...] --envs 2048
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_grasp_env as E  # noqa: E402


def main() -> None:
    import jax
    import jax.numpy as jp
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="+")
    ap.add_argument("--frame", default="flat_grasp.json")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    for xml in a.xml:
        env = E.make_mjx_env(xml, a.frame)
        reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
        t0 = time.time()
        s = reset(jax.random.split(jax.random.PRNGKey(0), a.envs))
        act = jax.random.uniform(jax.random.PRNGKey(1), (a.envs, env.action_size), minval=-1, maxval=1)
        s = step(s, act)
        jax.block_until_ready(s.obs)
        compile_s = time.time() - t0
        t0 = time.time()
        for _ in range(a.steps):
            s = step(s, act)
        jax.block_until_ready(s.obs)
        dt = time.time() - t0
        print(f"{xml}: compile {compile_s:.0f} s, {a.envs * a.steps / dt:,.0f} policy steps/s "
              f"(1 device, {a.envs} envs)", flush=True)


if __name__ == "__main__":
    main()
