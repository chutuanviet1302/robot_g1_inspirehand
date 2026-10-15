"""Simulate an episode in a separate process and return its states, for smooth playback in the viewer.

Live viewing stalls whenever the controller thinks (a 6-D pose fit ~0.25 s, planning a synchronised grasp up to a
few seconds): the scene freezes, then jumps. And a perception camera rendering off-screen in the viewer's process
crashed the viewer on Windows (two OpenGL contexts). So the episode runs first, in its own process, and the
viewer only plays the recorded joint and object positions back at real time.
"""

from __future__ import annotations

import os
import sys


def simulate(args: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    import numpy as np

    from homehand.runner import run_episode
    task = args["task"]
    if task == "conveyor":
        from homehand.control.conveyor_planner import ConveyorPlanner
        from homehand.env.conveyor_env import ConveyorEnv
        env = ConveyorEnv()
        controller = ConveyorPlanner(env, perception=args["perception"])
        reset = env.reset
        kw = args.get("reset_kw", {})
        env.reset = lambda task="conveyor", seed=0, objects=None, randomization=None, layout=None: reset(
            task, seed=seed, objects=objects, randomization=randomization, **kw)
    elif task in ("home", "home_conveyor"):
        from homehand.control.home_controller import HomeController
        from homehand.env.home_env import HomeConveyorEnv, HomeEnv
        env = HomeConveyorEnv() if task == "home_conveyor" else HomeEnv()
        controller = HomeController(env, perception=args["perception"])
    elif task == "bimanual_lift":
        from homehand.control.bimanual import BimanualExpert
        from homehand.env.bimanual_env import BimanualEnv
        env = BimanualEnv()
        controller = BimanualExpert(env)
    else:
        from homehand.control.planner import TidyPlanner
        from homehand.env.kitchen_env import KitchenEnv
        env = KitchenEnv()
        controller = TidyPlanner(env, perception=args["perception"])
    qpos = []
    r = run_episode(env, controller, task=task, seed=args["seed"], randomization=args.get("randomization"),
                    on_step=lambda obs, a: qpos.append(env.data.qpos.copy()))
    out = {"qpos": np.asarray(qpos), "outcome": r["outcome"], "object_outcomes": dict(r["object_outcomes"]),
           "objects": [str(o) for o in r["objects"]]}
    if task in ("home", "home_conveyor"):     # obstacles are mocap bodies with per-episode sizes: not in qpos
        m, d = env.model, env.data
        out["mocap"] = (d.mocap_pos.copy(), d.mocap_quat.copy())
        out["obstacle_geoms"] = {int(g): (m.geom_size[g].copy(), m.geom_pos[g].copy()) for g in env.obstacle_geoms}
        out["nav"] = env.info.details.get("nav")
    return out


def simulate_in_subprocess(args: dict) -> dict:
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(simulate, (args,))
