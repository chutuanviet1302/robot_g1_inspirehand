"""Run one episode of a controller in the kitchen environment (shared by CLI, data collection, eval and web)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from homehand.control.planner import TidyPlanner
from homehand.env.kitchen_env import KitchenEnv

SETTLE_STEPS = 15  # after the controller is done, let the scene settle before the final verdict


def run_episode(env: KitchenEnv, controller: TidyPlanner, task: str = "tidy_table", seed: int = 0,
                randomization=None, objects=None, layout=None,
                on_step: Callable[[dict, np.ndarray], bool | None] | None = None) -> dict:
    """Run until the task ends. `on_step(obs, action)` may return True to abort.

    Returns the episode summary (outcome, per-object outcomes, planner log, perception stats).
    """
    obs = env.reset(task, seed=seed, randomization=randomization, objects=objects, layout=layout)
    controller.reset(obs)
    done = False
    settle = 0
    perception_ms = []
    last_perc = None
    while not done:
        action = controller.act(obs)
        p = controller.last_perception
        if p is not None and p is not last_perc:
            perception_ms.append(p.latency_ms)
            last_perc = p
        obs, done = env.step(action)
        if on_step is not None and on_step(obs, action):
            env.force_finish()
            break
        if controller.done and not done:
            settle += 1
            if settle >= SETTLE_STEPS:
                env.force_finish()
                done = True
    info = env.info
    return {
        "task": task,
        "seed": seed,
        "objects": info.objects,
        "outcome": info.outcome,
        "success": info.success,
        "object_outcomes": info.object_outcomes,
        "n_cleared": info.n_cleared,
        "n_objects": len(info.objects),
        "steps": info.steps,
        "sim_time": round(info.steps * 0.04, 2),
        "randomization": info.randomization,
        "sampled": info.details.get("sampled", {}),
        "perception": controller.perception_mode,
        "skill": controller.skill_kind,
        "perception_ms": round(float(np.mean(perception_ms)), 1) if perception_ms else 0.0,
        "log": _jsonable(controller.log),
        "events": _jsonable(env.events + env.safety.stats.events),
        "safety": _jsonable(env.safety.stats.to_dict()),
        "layout": [list(map(lambda v: v if isinstance(v, str) else round(float(v), 4), l)) for l in env.layout],
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.str_):
        return str(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x
