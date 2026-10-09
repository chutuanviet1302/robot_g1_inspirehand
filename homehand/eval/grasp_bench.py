"""Grasp benchmark: the pick-and-place skill on one object at a time, over a grid of positions and yaws.

    homehand grasp-bench                       # every object, both hands, inner + outer slot, 6 yaws
    homehand grasp-bench --objects meat_can -n 12

Each trial is a `pick_place` episode with ground-truth perception, so it isolates the skill from perception and
planning. Besides the episode outcome it records *where* a failure happened:

  missed      the fingers closed on nothing (empty-grasp check fired)
  pushed      the object moved > 2 cm before the fingers closed (swept by the approaching hand)
  slipped     the object was lifted, then fell out of the hand before the drop point
  bounced     released over the bin but it did not stay inside
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

from homehand.env.kitchen_env import SPAWN_ZONES
from homehand.model.objects import OBJECTS

_W: dict = {}


def trials(objects: list[str], n_yaw: int, seed: int = 0, rests=("upright", "lying")) -> list[tuple]:
    from homehand.env.kitchen_env import LONG_LYING, LONG_LYING_ABSY, LYING_X, LYING_YAW_JITTER, SHORT_LYING_YAW
    rng = np.random.default_rng(seed)
    out = []
    for name in objects:
        for rest in rests:
            for side, sgn in (("left", 1.0), ("right", -1.0)):
                for slot, z in SPAWN_ZONES.items():
                    if rest == "lying" and slot == "inner":
                        continue                       # lying objects only spawn in the outer slots
                    for k in range(n_yaw):
                        x = float(rng.uniform(*z["x"]))
                        y = float(sgn * rng.uniform(*z["absy"]))
                        yaw = float(2 * np.pi * k / n_yaw + rng.uniform(-0.2, 0.2))
                        if rest == "lying":
                            x = float(rng.uniform(*LYING_X))
                        if rest == "lying" and name in LONG_LYING:
                            y = float(sgn * rng.uniform(*LONG_LYING_ABSY))
                            yaw = float(np.pi / 2 + rng.uniform(-LYING_YAW_JITTER, LYING_YAW_JITTER))
                        elif rest == "lying":
                            c, w = SHORT_LYING_YAW[name]
                            yaw = float(c - w + 2 * w * (k + 0.5) / n_yaw)
                        out.append((name, side, f"{slot}/{rest}", x, y, yaw, rest))
    return out


def _init() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    env = KitchenEnv()
    _W["env"], _W["planner"] = env, TidyPlanner(env, perception="ground_truth")


def run_trial(trial: tuple) -> dict:
    from homehand.runner import run_episode
    name, side, slot, x, y, yaw, rest = trial
    env, planner = _W["env"], _W["planner"]
    st = {"start": None, "closed_pos": None, "release_phase": None, "lifted": False}

    def on_step(obs, action):
        sk = planner.skill
        pos = obs["objects"][name]["pos"]
        if st["start"] is None:
            st["start"] = obs["objects"][name]["centre"].copy()
        pos = obs["objects"][name]["centre"]
        if sk is not None and sk.phase == "grasp" and st["closed_pos"] is None:
            st["closed_pos"] = pos.copy()
        tr = env.tracks[name]
        st["lifted"] |= tr.grasped_once
        if tr.grasped_once and not tr.in_hand and st["release_phase"] is None:
            st["release_phase"] = sk.phase if sk is not None else "done"

    ep = run_episode(env, planner, task="pick_place", seed=0, layout=[(name, x, y, yaw, rest)], on_step=on_step)
    outcome = ep["object_outcomes"][name]
    statuses = [l.get("status") for l in ep["log"] if l["event"] == "skill_done"]
    if outcome == "success":
        mode = "ok"
    elif st["closed_pos"] is not None and np.linalg.norm(st["closed_pos"][:2] - st["start"][:2]) > 0.02 \
            and not st["lifted"]:
        mode = "pushed"
    elif st["lifted"] and st["release_phase"] not in ("drop", "return", "done", None):
        mode = "slipped"
    elif st["lifted"]:
        mode = "bounced"
    elif "empty_grasp" in statuses:
        mode = "missed"
    else:
        mode = outcome
    return {"object": name, "side": side, "slot": slot, "x": x, "y": y, "yaw": yaw, "outcome": outcome,
            "mode": mode, "release_phase": st["release_phase"], "attempts": len(statuses)}


def run_bench(objects: list[str] | None = None, n_yaw: int = 6, workers: int | None = None,
              seed: int = 0, progress=None, rests=("upright", "lying")) -> list[dict]:
    from homehand.resources import safe_workers
    todo = trials(objects or list(OBJECTS), n_yaw, seed, rests)
    n = safe_workers(workers)
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n, initializer=_init) as pool:
        for r in pool.imap_unordered(run_trial, todo):
            results.append(r)
            if progress:
                progress(len(results), len(todo))
    return results


def summarize(results: list[dict]) -> str:
    """Success per object / side / slot and the failure modes, as a Markdown table."""
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r["object"], r["side"], r["slot"])].append(r)
    lines = ["| object | hand | slot | success | failures |", "|---|---|---|---|---|"]
    for (obj, side, slot), rs in sorted(groups.items()):
        ok = sum(r["mode"] == "ok" for r in rs)
        modes = defaultdict(int)
        for r in rs:
            if r["mode"] != "ok":
                modes[r["mode"]] += 1
        fails = ", ".join(f"{m} {c}" for m, c in sorted(modes.items())) or "-"
        lines.append(f"| {obj} | {side} | {slot} | {ok}/{len(rs)} | {fails} |")
    tot = sum(r["mode"] == "ok" for r in results)
    lines.append(f"| **all** | | | **{tot}/{len(results)}** | |")
    return "\n".join(lines)
