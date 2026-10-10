"""Record pick-and-place demonstrations from the scripted expert for imitation learning.

    homehand collect --name expert_v1 -n 300 --perception oracle --randomization low

Every worker runs whole `tidy_table` episodes with the expert planner in collect mode; each skill execution
(one hand, one object: reach -> grasp -> carry -> drop -> return) is one demonstration of (39-dim features,
13-dim action) pairs. Only *successful* executions are kept: the skill reported `placed` and the object ended
the episode in the bin (a drop that bounced out, or a first attempt that was followed by a retry, is not a
demonstration of the task).

Layout on disk (`data/datasets/<name>/`):

    shards/seed00042.npz   one file per episode (resumable: finished seeds are skipped after a reboot)
    data.npz               obs (N, 39) float32, action (N, 13) float32, episode_index (N,) int32 (one index per demo)
    meta.json              counts, per object / per hand, demo lengths, normalisation stats, feature names
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import numpy as np

from homehand import paths

_W: dict = {}
STD_FLOOR = 1e-2   # constant-ish features (e.g. a rare grasp one-hot) must not blow up after normalisation


def dataset_dir(name: str):
    return paths.DATASET_DIR / name


def _init_worker(perception: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    env = KitchenEnv()
    _W["env"] = env
    _W["planner"] = TidyPlanner(env, perception=perception, collect=True)


def _run_one(args) -> dict:
    """One tidy episode -> its successful demonstrations, written to a shard file."""
    shard, seed, randomization = args
    from homehand.runner import run_episode
    env, planner = _W["env"], _W["planner"]
    ep = run_episode(env, planner, task="tidy_table", seed=seed, randomization=randomization)
    demos = planner.demos
    # the last execution per object is the one that decided its outcome
    last = {d["object"]: i for i, d in enumerate(demos)}
    keep = [d for i, d in enumerate(demos)
            if d["status"] == "placed" and last[d["object"]] == i and ep["object_outcomes"].get(d["object"]) == "success"]
    arrays = {"n": np.array(len(keep))}
    for k, d in enumerate(keep):
        arrays[f"obs{k}"] = np.asarray(d["obs"], np.float32)
        arrays[f"act{k}"] = np.asarray(d["action"], np.float32)
        arrays[f"side{k}"] = np.array(d["side"])
        arrays[f"object{k}"] = np.array(d["object"])
    tmp = shard.with_name(shard.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(shard)
    return {"seed": seed, "outcome": ep["outcome"], "n_executions": len(demos), "n_kept": len(keep)}


def collect(name: str = "expert_v1", n_episodes: int = 400, workers: int | None = None, perception: str = "oracle",
            randomization: str = "low", seed0: int = 0, progress=None) -> dict:
    """Run `n_episodes` expert episodes (parallel, RAM-guarded) and merge the kept demos into one dataset."""
    from homehand.resources import WORKER_MB, WORKER_TORCH_MB, safe_workers
    out = dataset_dir(name)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    jobs = [(shards / f"seed{seed0 + i:05d}.npz", seed0 + i, randomization) for i in range(n_episodes)]
    todo = [j for j in jobs if not j[0].exists()]
    if len(todo) < len(jobs):
        print(f"[collect] {len(jobs) - len(todo)} episodes already recorded, {len(todo)} to go")
    t0 = time.time()
    if todo:
        n = safe_workers(workers, per_worker_mb=WORKER_TORCH_MB if perception == "yolo" else WORKER_MB)
        ctx = mp.get_context("spawn")
        with ctx.Pool(n, initializer=_init_worker, initargs=(perception,)) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_one, todo), 1):
                if progress:
                    progress(i, len(todo))
                elif i % 10 == 0 or i == len(todo):
                    print(f"[collect] {i}/{len(todo)} episodes ({time.time() - t0:.0f}s), "
                          f"last: seed {r['seed']} {r['outcome']} kept {r['n_kept']}/{r['n_executions']}")
    meta = merge(name, [j[0] for j in jobs], perception=perception, randomization=randomization)
    meta["collect_seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] {name}: {meta['n_demos']} demos / {meta['n_frames']} frames from {meta['n_tidy_episodes']} episodes")
    return meta


def merge(name: str, shard_files, perception: str = "oracle", randomization: str = "low") -> dict:
    """Concatenate shards into data.npz and compute the normalisation statistics."""
    from homehand.policy.features import ACT_DIM, FEATURE_NAMES, OBS_DIM
    obs, act, epi, lengths = [], [], [], []
    per_object: dict[str, int] = {}
    per_side = {"left": 0, "right": 0}
    k = 0
    for f in sorted(shard_files):
        with np.load(f) as z:
            for i in range(int(z["n"])):
                o, a = z[f"obs{i}"], z[f"act{i}"]
                obs.append(o)
                act.append(a)
                epi.append(np.full(len(o), k, np.int32))
                lengths.append(len(o))
                obj, side = str(z[f"object{i}"]), str(z[f"side{i}"])
                per_object[obj] = per_object.get(obj, 0) + 1
                per_side[side] += 1
                k += 1
    if not obs:
        raise RuntimeError(f"dataset {name}: no successful demonstrations recorded")
    obs_a, act_a = np.concatenate(obs), np.concatenate(act)
    assert obs_a.shape[1] == OBS_DIM and act_a.shape[1] == ACT_DIM, (obs_a.shape, act_a.shape)
    out = dataset_dir(name)
    np.savez(out / "data.npz", obs=obs_a, action=act_a, episode_index=np.concatenate(epi))
    lengths = np.asarray(lengths)
    return {
        "name": name, "n_demos": int(k), "n_frames": int(len(obs_a)), "n_tidy_episodes": len(list(shard_files)),
        "perception": perception, "randomization": randomization,
        "per_object": dict(sorted(per_object.items())), "per_side": per_side,
        "demo_length": {"min": int(lengths.min()), "mean": round(float(lengths.mean()), 1), "max": int(lengths.max())},
        "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "feature_names": FEATURE_NAMES,
        "stats": {
            "obs_mean": obs_a.mean(0).tolist(), "obs_std": np.maximum(obs_a.std(0), STD_FLOOR).tolist(),
            "act_mean": act_a.mean(0).tolist(), "act_std": np.maximum(act_a.std(0), STD_FLOOR).tolist(),
        },
    }


def load(name: str) -> tuple[dict, dict]:
    """(arrays, meta) of a recorded dataset; `name` may also be a path to a dataset directory."""
    from pathlib import Path
    d = Path(name) if Path(name).is_dir() else dataset_dir(name)
    if not (d / "data.npz").exists():
        raise FileNotFoundError(f"{d / 'data.npz'} not found: run `homehand collect --name {name}` first")
    meta = json.loads((d / "meta.json").read_text())
    with np.load(d / "data.npz") as z:
        data = {k: z[k] for k in ("obs", "action", "episode_index")}
    return data, meta
