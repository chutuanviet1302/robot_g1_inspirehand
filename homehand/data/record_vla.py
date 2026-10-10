"""Record image + language + action demonstrations for a vision-language-action policy (SmolVLA).

    homehand collect-vla --name vla_v1 -n 120

Each successful pick-and-place of the expert is one episode: the three camera streams (head, left wrist, right
wrist; 256x256 at 25 Hz) as mp4, the 26-dim joint state and the 26-dim action, and the instruction, e.g.
"pick up the mustard bottle with the left hand and put it in the bin". The task planner stays the high level
(which object, which hand); the VLA learns the skill from pixels + proprioception + language.

Raw layout (`data/datasets_vla/<name>/`): `seedNNNNN.json` per tidy episode (resumable) listing its demos,
`seedNNNNN_k.npz` (state, action) and `seedNNNNN_k_<camera>.mp4`. `homehand/data/export_lerobot.py` (run in the
LeRobot environment) turns this into a LeRobotDataset.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import numpy as np

from homehand import paths
from homehand.model import spec

VLA_DIR = paths.DATA_DIR / "datasets_vla"
IMG = 256
FPS = round(1 / spec.CONTROL_DT)
OBJECT_PHRASE = {"soup_can": "tomato soup can", "mustard": "mustard bottle", "sugar_box": "sugar box",
                 "meat_can": "potted meat can"}
TEMPLATES = [
    "pick up the {obj} with the {side} hand and put it in the bin",
    "use your {side} hand to put the {obj} in the bin",
    "{side} hand: move the {obj} into the bin",
    "grab the {obj} with your {side} hand and drop it into the bin",
]
HELD_OUT_TEMPLATE = "with the {side} hand, tidy the {obj} away into the bin"   # never in training: language test
_W: dict = {}


def instruction(obj: str, side: str, k: int = 0, held_out: bool = False) -> str:
    tpl = HELD_OUT_TEMPLATE if held_out else TEMPLATES[k % len(TEMPLATES)]
    return tpl.format(obj=OBJECT_PHRASE[obj], side=side)


def _init_worker(perception: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    env = KitchenEnv()
    _W["env"], _W["planner"] = env, TidyPlanner(env, perception=perception)


def render_cameras(env, size: int = IMG) -> dict[str, np.ndarray]:
    r = env.renderer(size, size)
    out = {}
    for cam in spec.CAMERAS_VLA:
        r.update_scene(env.data, camera=cam)
        out[cam] = r.render().copy()
    return out


def state_vector(obs: dict) -> np.ndarray:
    return np.concatenate([obs["arm_q"], obs["hand_q"]]).astype(np.float32)


def _run_one(args) -> dict:
    out_dir, seed, randomization = args
    import imageio.v2 as imageio

    from homehand.runner import run_episode
    env, planner = _W["env"], _W["planner"]
    execs: list[dict] = []

    def close(e):
        for w in e["writers"].values():
            w.close()
        e["writers"] = {}

    def step_hook(obs, action, current):
        if not execs or execs[-1].get("status") is not None:
            j = len(execs)
            # frames go straight into the video files (holding them would take ~1 GB per episode)
            writers = {c: imageio.get_writer(out_dir / f"seed{seed:05d}_x{j}_{c}.mp4", fps=FPS, quality=8,
                                             macro_block_size=8) for c in spec.CAMERAS_VLA}
            execs.append({"j": j, "side": current["side"], "object": current["target"]["name"], "state": [],
                          "action": [], "writers": writers, "status": None})
        e = execs[-1]
        e["state"].append(state_vector(obs))
        e["action"].append(np.asarray(action, np.float32))
        for cam, img in render_cameras(env).items():
            e["writers"][cam].append_data(img)

    def skill_hook(current, status):
        if execs:
            execs[-1]["status"] = status
            close(execs[-1])

    planner.step_hook, planner.skill_hook = step_hook, skill_hook
    try:
        ep = run_episode(env, planner, task="tidy_table", seed=seed, randomization=randomization)
    finally:
        planner.step_hook = planner.skill_hook = None
        for e in execs:
            close(e)
    last = {e["object"]: i for i, e in enumerate(execs)}
    demos = []
    for i, e in enumerate(execs):
        keep = e["status"] == "placed" and last[e["object"]] == i and             ep["object_outcomes"].get(e["object"]) == "success"
        for cam in spec.CAMERAS_VLA:
            src = out_dir / f"seed{seed:05d}_x{e['j']}_{cam}.mp4"
            if keep:
                src.replace(out_dir / f"seed{seed:05d}_{len(demos)}_{cam}.mp4")
            else:
                src.unlink(missing_ok=True)
        if not keep:
            continue
        stem = f"seed{seed:05d}_{len(demos)}"
        np.savez_compressed(out_dir / f"{stem}.npz", state=np.stack(e["state"]), action=np.stack(e["action"]))
        k = seed * 7 + len(demos)
        demos.append({"stem": stem, "object": str(e["object"]), "side": e["side"], "frames": len(e["state"]),
                      "instruction": instruction(str(e["object"]), e["side"], k)})
    meta = {"seed": seed, "outcome": ep["outcome"], "executions": len(execs), "demos": demos}
    tmp = out_dir / f"seed{seed:05d}.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(out_dir / f"seed{seed:05d}.json")          # written last: the seed counts as done
    return meta


def collect(name: str = "vla_v1", n_episodes: int = 120, workers: int | None = None, perception: str = "oracle",
            randomization: str = "low", seed0: int = 0) -> dict:
    from homehand.resources import safe_workers
    out = VLA_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(out, seed0 + i, randomization) for i in range(n_episodes)
            if not (out / f"seed{seed0 + i:05d}.json").exists()]
    print(f"[collect-vla] {n_episodes - len(jobs)} episodes already recorded, {len(jobs)} to go")
    t0 = time.time()
    if jobs:
        n = safe_workers(workers, per_worker_mb=1400)       # renderer + three video buffers per worker
        with mp.get_context("spawn").Pool(n, initializer=_init_worker, initargs=(perception,)) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_one, jobs), 1):
                if i % 5 == 0 or i == len(jobs):
                    print(f"[collect-vla] {i}/{len(jobs)} ({time.time() - t0:.0f}s), seed {r['seed']}: "
                          f"{len(r['demos'])}/{r['executions']} kept", flush=True)
    metas = [json.loads(p.read_text()) for p in sorted(out.glob("seed*.json"))]
    demos = [d for m in metas for d in m["demos"]]
    summary = {"name": name, "episodes": len(metas), "demos": len(demos), "frames": sum(d["frames"] for d in demos),
               "fps": FPS, "image_size": IMG, "cameras": list(spec.CAMERAS_VLA), "state_dim": spec.ACTION_DIM,
               "action_dim": spec.ACTION_DIM, "state_names": list(spec.ACTUATORS),
               "templates": TEMPLATES, "held_out_template": HELD_OUT_TEMPLATE,
               "perception": perception, "randomization": randomization}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[collect-vla] {name}: {summary['demos']} demos / {summary['frames']} frames from {summary['episodes']} episodes")
    return summary
