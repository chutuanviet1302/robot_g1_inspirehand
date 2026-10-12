"""ROS 2 bridge logic without ROS: the message conversions, and an episode passed through them (observation as
JSON, command as joint-name / position lists, the step tag) gives exactly the offline episode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from homehand import paths

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "homehand_ros"))
pytestmark = pytest.mark.skipif(not paths.SCENE_XML.exists(), reason="run `homehand build-model` first")


def test_command_round_trip_and_rejects_bad_commands():
    from homehand_ros.convert import JOINT_NAMES, action_from_command, command_tag, parse_tag
    a = np.linspace(-1, 1, 26)
    perm = np.random.default_rng(0).permutation(26)
    names = [JOINT_NAMES[i] for i in perm]
    assert np.array_equal(action_from_command(names, list(a[perm])), a)          # any joint order
    assert action_from_command(names[:-1], list(a[perm][:-1])) is None             # a joint missing
    bad = a.copy()
    bad[3] = np.nan
    assert action_from_command(JOINT_NAMES, list(bad)) is None                     # not finite
    assert parse_tag(command_tag(17, False)) == (17, False)
    assert parse_tag(command_tag(17, True)) == (17, True)
    assert parse_tag("garbage") == (-1, False)


def test_episode_through_the_messages_equals_the_offline_episode():
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.control.planner import TidyPlanner
    from homehand.runner import SETTLE_STEPS, run_episode
    from homehand_ros.convert import (JOINT_NAMES, action_from_command, command_positions, command_tag,
                                      obs_from_json, obs_to_json, parse_tag)
    from homehand_ros.controllers import make_controller

    layout = [("soup_can", 0.31, -0.18, 0.0)]
    env = KitchenEnv()
    offline = []
    ref = run_episode(env, TidyPlanner(env, perception="ground_truth"), task="pick_place", seed=0, layout=layout,
                      on_step=lambda obs, a: offline.append(np.array(a)))

    # the sim bridge's loop (lockstep), with the controller node's work in between
    env = KitchenEnv()
    obs = env.reset("pick_place", seed=0, layout=layout)
    ctrl = make_controller("expert", "pick_place")
    ctrl.reset(obs_from_json(obs_to_json(obs)))
    sent, settle, controller_done, done = [], SETTLE_STEPS, False, False
    while not done:
        o = obs_from_json(obs_to_json(obs))                       # /homehand/observation
        a = ctrl.act(o)
        tag = command_tag(int(o["t"]), bool(ctrl.done))           # /joint_command
        action = action_from_command(JOINT_NAMES, command_positions(a))
        step, finished = parse_tag(tag)
        assert step == obs["t"]
        controller_done |= finished
        sent.append(action)
        obs, done = env.step(action)
        if controller_done and not done:
            settle -= 1
            if settle <= 0:
                env.force_finish()
                done = True
    assert env.info.outcome == ref["outcome"] == "success"
    assert len(sent) == len(offline) and np.array_equal(np.array(sent), np.array(offline))
