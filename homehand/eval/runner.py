"""Batch evaluation: N episodes in parallel worker processes, results in SQLite + replay files.

    homehand eval --controller expert --perception yolo --randomization medium -n 100
    homehand eval --controller act_expert_v1 ...
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Callable

import numpy as np

from homehand import paths
from homehand.eval import store
from homehand.eval.metrics import summarize

_W: dict = {}


def _init_worker(controller: str, perception: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
    import torch

    torch.set_num_threads(1)
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv

    policy = None
    if controller != "expert":
        from homehand.policy.skill import load_policy
        policy = load_policy(controller, device="cpu")  # many small workers: CPU inference
    env = KitchenEnv(record=True)
    _W["env"] = env
    _W["planner"] = TidyPlanner(env, perception=perception, policy=policy)


def _run_one(args) -> tuple[dict, str]:
    run_id, seed, task, randomization = args
    from homehand.runner import run_episode

    env, planner = _W["env"], _W["planner"]
    ep = run_episode(env, planner, task=task, seed=seed, randomization=randomization)
    replay = paths.EPISODE_DIR / f"run{run_id}" / f"seed{seed}.npz"
    replay.parent.mkdir(parents=True, exist_ok=True)
    arr = env.episode_arrays()
    np.savez_compressed(replay, qpos=arr["qpos"].astype(np.float32))
    return ep, str(replay.relative_to(paths.DATA_DIR))


def run_eval(controller: str = "expert", perception: str = "auto", randomization: str = "nominal",
             n: int = 50, task: str = "tidy_table", seed0: int = 0, workers: int | None = None,
             name: str | None = None, progress: Callable[[int, int], None] | None = None,
             run_id: int | None = None) -> int:
    from homehand.perception.detector import YOLO_WEIGHTS

    if perception == "auto":
        perception = "yolo" if YOLO_WEIGHTS.exists() else "oracle"
    from homehand.resources import WORKER_MB, WORKER_TORCH_MB, safe_workers
    heavy = perception == "yolo" or controller != "expert"   # PyTorch in every worker
    # never more workers than the free RAM can hold
    workers = safe_workers(workers, per_worker_mb=WORKER_TORCH_MB if heavy else WORKER_MB)
    name = name or f"{controller} | {perception} | {randomization}"
    if run_id is None:
        run_id = store.create_run(name, task, controller, perception, randomization, n, seed0)
    t0 = time.time()
    episodes = []
    ctx = mp.get_context("spawn")
    jobs = [(run_id, seed0 + i, task, randomization) for i in range(n)]
    try:
        with ctx.Pool(workers, initializer=_init_worker, initargs=(controller, perception)) as pool:
            for ep, replay in pool.imap_unordered(_run_one, jobs):
                episodes.append(ep)
                store.add_episode(run_id, ep, replay)
                if progress:
                    progress(len(episodes), n)
        summary = summarize(episodes)
        store.finish_run(run_id, summary, time.time() - t0)
    except Exception:
        store.finish_run(run_id, summarize(episodes) if episodes else {}, time.time() - t0, status="failed")
        raise
    return run_id
