"""Gymnasium environment for the bimanual grip policy (`homehand.rl.grip`): one box lift per episode.

The agent decides every DECIDE_EVERY control steps while the box is squeezed, lifted, carried and put down;
the rest of the plan runs by itself. Physics randomisation per episode (default `high`: friction x0.5-1.5,
mass x0.5-2, latency, finger gains), so the policy has to infer from touch and arm sag how hard to press.

Reward (logged per term in `info["terms"]`):
  lift      potential shaping on the box height while both hands touch it (saturates at the lift height)
  success   +SUCCESS_BONUS at the end if the task succeeded (held, carried, put down upright at the goal)
  drop      -DROP_PENALTY and the episode ends as soon as the box is lost
  fail      -FAIL_PENALTY at the end for not lifting / misplacing it
  force     -FORCE_W per newton of total finger force above FORCE_FREE per hand (do not crush)
  smooth    -SMOOTH_W * |a - a_prev|^2
  stop      -STOP_PENALTY on a protective stop
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from homehand.control.bimanual import LIFT_H
from homehand.env.bimanual_env import BimanualEnv
from homehand.env.kitchen_env import TABLE_Z
from homehand.rl.grip import ACT_DIM, ACTIVE, DECIDE_EVERY, OBS_DIM, SIDES, GripController

SUCCESS_BONUS = 10.0
DROP_PENALTY = 10.0
FAIL_PENALTY = 5.0
LIFT_W = 2.0
FORCE_FREE = 60.0
FORCE_W = 0.002
SMOOTH_W = 0.05
STOP_PENALTY = 10.0
SETTLE = 15


class GripEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, randomization: str = "high", seed: int = 0):
        self.env = BimanualEnv()
        self.ctrl = GripController(self.env)
        self.randomization = randomization
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    def _phase(self) -> str:
        return self.ctrl.plan[min(self.ctrl.k, len(self.ctrl.plan) - 1)][0]

    def _run(self, n: int | None = None) -> None:
        """Step the plan `n` times, or until an active phase / the end when n is None."""
        i = 0
        while not self.ctrl.done and not self.done and (n is None and self._phase() not in ACTIVE or i < (n or 0)):
            self.obs, self.done = self.env.step(self.ctrl.act(self.obs))
            i += 1

    def _phi(self) -> float:
        held = self.env.hands_on_tote() == set(SIDES)
        lift = float(self.env.tote_pose()[0][2] - TABLE_Z)
        return LIFT_W * min(max(lift, 0.0), LIFT_H) / LIFT_H if held else 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.obs = self.env.reset(seed=int(self.rng.integers(1_000_000, 2_000_000)), randomization=self.randomization)
        self.ctrl.reset(self.obs)
        self.done = False
        self._run()
        self.phi = self._phi()
        return self.ctrl.observation(self.obs), {}

    def step(self, a):
        terms = dict.fromkeys(("lift", "success", "drop", "fail", "force", "smooth", "stop"), 0.0)
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        terms["smooth"] = -SMOOTH_W * float(np.sum((a - self.ctrl.prev) ** 2))
        self.ctrl.set_action(a)
        self._run(DECIDE_EVERY)
        f = [float(np.sum(self.obs["finger_force"][s])) for s in SIDES]
        terms["force"] = -FORCE_W * sum(max(0.0, x - FORCE_FREE) for x in f)
        phi = self._phi()
        terms["lift"] = phi - self.phi
        self.phi = phi
        over = False
        if self.env.dropped:
            terms["drop"], over = -DROP_PENALTY, True
        elif self.env.safety.stats.protective_stop:
            terms["stop"], over = -STOP_PENALTY, True
        elif self._phase() not in ACTIVE:           # held phases over: finish the plan, judge the result
            self._run(len(self.ctrl.plan))
            for _ in range(SETTLE):
                if self.done:
                    break
                self.obs, self.done = self.env.step(self.env.last_action)
            self.env.force_finish()
            out = self.env.info.outcome
            terms["success" if out == "success" else "fail"] = SUCCESS_BONUS if out == "success" else -FAIL_PENALTY
            over = True
        if over and self.env.info.outcome == "running":
            self.env.force_finish()
        info = {"terms": terms, "carried": self.env.info.outcome == "success", "outcome": self.env.info.outcome}
        obs = self.ctrl.observation(self.obs) if not over else np.zeros(OBS_DIM, np.float32)
        return obs, float(sum(terms.values())), over, False, info
