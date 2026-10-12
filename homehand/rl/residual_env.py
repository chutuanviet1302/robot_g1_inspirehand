"""Gymnasium environment for the residual policy: one pick of a real tidy-table scene per episode.

reset: a tidy-table layout (random seed, physics randomisation), the expert first clears 0-1 objects (1 in 4) (so the
policy sees cluttered and sparse counters), then the planner chooses the next object and hand as usual.
The expert runs every non-contact phase by itself; the agent is asked only in `ACTIVE_PHASES`. The episode
ends when the hand has carried the object to the release point (the drop, the retreat and the way back cannot
be changed by the residual and are not simulated: 2-3x more episodes per hour), after an empty grasp, when the
object or a neighbour falls, or on a protective stop.

Reward (every term is logged in `info["terms"]`):
  lift      potential shaping on the target's height while it is in the hand (saturates at LIFT_FULL)
  lifted    +LIFTED_BONUS once the lift phase ends with the object in the hand
  carried   +CARRY_BONUS if the object is still in the hand at the release point
  neighbour -NEIGHBOUR_PENALTY per other object nudged (moved > 1 cm / tilted > 10 deg) during the pick,
            again if it tips over
  target    -TARGET_PENALTY if the target itself tips over / falls before it was lifted
  force     -FORCE_W per newton above FORCE_FREE on any finger (expert grasps: thumb 25-35 N, fingers 5-10 N;
            only spikes beyond that are penalised)
  action    -ACTION_W * |a|^2 (stay close to the expert)
  stop      -STOP_PENALTY and terminate on a protective stop of the safety layer
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from homehand.control.planner import TidyPlanner
from homehand.env.kitchen_env import TABLE_Z, KitchenEnv
from homehand.rl.residual import ACT_DIM, ACTIVE_PHASES, OBS_DIM, apply_residual, residual_obs

LIFT_FULL = 0.05
LIFT_W = 2.0
LIFTED_BONUS = 3.0
CARRY_BONUS = 10.0
NEIGHBOUR_PENALTY = 5.0
TARGET_PENALTY = 5.0
FORCE_FREE = 40.0
FORCE_W = 0.01
ACTION_W = 0.01
STOP_PENALTY = 10.0
MAX_AGENT_STEPS = 150
MAX_AUTO_STEPS = 600


class ResidualPickEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, randomization: str = "medium", seed: int = 0, neighbour_penalty: float = NEIGHBOUR_PENALTY):
        self.env = KitchenEnv()
        self.planner = TidyPlanner(self.env, perception="ground_truth")
        self.randomization = randomization
        self.neighbour_penalty = neighbour_penalty
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    # ------------------------------------------------------------------ helpers
    @property
    def skill(self):
        return self.planner.skill

    def _step_env(self, action: np.ndarray) -> None:
        sk = self.skill
        self.env.context = {"phase": f"{sk.side}:{sk.phase}", "target": sk.target.name}
        self.obs, self.env_done = self.env.step(action)

    def _lift(self) -> float:
        tr = self.env.tracks[self.target]
        return float(self.env.object_pose(self.target)[0][2] - TABLE_Z) if tr.in_hand else 0.0

    def _auto(self) -> None:
        """Let the expert run until the agent is needed again or the pick is over."""
        n = 0
        while not self._over() and self.skill.phase not in ACTIVE_PHASES and n < MAX_AUTO_STEPS:
            self._step_env(self.skill.act(self.obs))
            n += 1

    def _over(self) -> bool:
        return (self.skill.done or self.skill.phase == "drop" or self.env_done
                or self.env.safety.stats.protective_stop)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        while True:
            ep_seed = int(self.rng.integers(1_000_000, 2_000_000))   # never the evaluation seeds (0-99)
            self.obs = self.env.reset("tidy_table", seed=ep_seed, randomization=self.randomization)
            self.planner.reset(self.obs)
            self.env_done = False
            skip, picks = int(self.rng.random() < 0.25), 0
            while not self.env_done and not self.planner.done:
                a = self.planner.act(self.obs)          # starts the next pick when it chooses an object
                if self.planner.state == "execute" and self.skill.steps == 1:
                    if picks == skip:
                        break
                    picks += 1
                self.obs, self.env_done = self.env.step(a)
            if self.planner.state == "execute" and not self.env_done:
                self.obs, self.env_done = self.env.step(a)
                break                                   # a pick has started: the episode is that pick
        self.target = self.skill.target.name
        self.others0 = {n: (tr.disturbed is not None, tr.tipped) for n, tr in self.env.tracks.items()
                        if n != self.target}
        self.prev = np.zeros(ACT_DIM, np.float32)
        self.agent_steps = 0
        self.lift_bonus_given = False
        self._auto()
        self.phi = LIFT_W * min(self._lift(), LIFT_FULL) / LIFT_FULL
        if self._over():                                  # nothing for the agent to do: start again
            return self.reset()
        return residual_obs(self.obs, self.skill, self.prev), {}

    def step(self, a):
        a = np.clip(np.asarray(a, np.float32), -1.0, 1.0)
        terms = dict.fromkeys(("lift", "lifted", "carried", "neighbour", "target", "force", "action", "stop"), 0.0)
        phase_was = self.skill.phase
        base = self.skill.act(self.obs)
        self._step_env(apply_residual(base, self.skill.side, a))
        self.prev = a
        self.agent_steps += 1
        terms["action"] = -ACTION_W * float(a @ a)
        f = np.asarray(self.obs["finger_force"][self.skill.side])
        terms["force"] = -FORCE_W * float(np.maximum(0.0, f - FORCE_FREE).sum())
        tr = self.env.tracks[self.target]
        # judged right after the agent's step, before the expert carries on by itself
        phi = LIFT_W * min(self._lift(), LIFT_FULL) / LIFT_FULL
        terms["lift"] = phi - self.phi
        self.phi = phi
        if not self.lift_bonus_given and phase_was == "lift" and self.skill.phase != "lift":
            self.lift_bonus_given = True
            terms["lifted"] = LIFTED_BONUS if tr.in_hand and tr.grasped_once else 0.0
        self._auto()
        for n, (was_dist, was_tip) in self.others0.items():
            o = self.env.tracks[n]
            if o.disturbed is not None and not was_dist:
                terms["neighbour"] -= self.neighbour_penalty
            if o.tipped and not was_tip:
                terms["neighbour"] -= self.neighbour_penalty
            self.others0[n] = (o.disturbed is not None, o.tipped)
        failed = tr.tipped or (tr.disturbed is not None and tr.disturbed["was_target"]
                               and tr.disturbed["tilt_deg"] > 30 and not tr.grasped_once)
        if failed:
            terms["target"] = -TARGET_PENALTY
        stop = bool(self.env.safety.stats.protective_stop)
        if stop:
            terms["stop"] = -STOP_PENALTY
        over = self._over() or failed
        if over and not stop and self.skill.phase == "drop" and tr.in_hand:
            terms["carried"] = CARRY_BONUS
        truncated = not over and self.agent_steps >= MAX_AGENT_STEPS
        info = {"terms": terms, "carried": terms["carried"] > 0, "object": self.target}
        obs = residual_obs(self.obs, self.skill, self.prev) if not over else np.zeros(OBS_DIM, np.float32)
        return obs, float(sum(terms.values())), bool(over), bool(truncated), info
