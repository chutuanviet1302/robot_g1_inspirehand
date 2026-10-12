"""RL for the bimanual box lift: a policy sets how hard each palm presses on the box, from touch and proprioception.

The scripted expert (`control.bimanual`) presses with a fixed squeeze (1.8 cm past the box's sides); with a heavy
or slippery box it loses it (16 % of the episodes at the `medium` physics randomisation, mostly dropped while
carried). Here the squeeze of each hand is chosen while the box is held. Two complete plans are made per
episode, with the palms SQUEEZE_LO and SQUEEZE_HI past the sides and the same timing; the command of each arm
interpolates between them with the policy's alpha (expert: ALPHA_EXPERT), so nothing is solved while the arms move.

Observation (sensors a real robot has): the 5 finger forces of each hand, how far each measured palm lags the
commanded one (a heavy box sags the position-servoed arms), the box's height above the counter and tilt (head
camera), the phase and the progress through it, the previous action.
"""

from __future__ import annotations

import numpy as np

from homehand.control.bimanual import SQUEEZE, BimanualExpert
from homehand.control.skills import compose_action
from homehand.env.kitchen_env import TABLE_Z

SQUEEZE_LO, SQUEEZE_HI = 0.0, 0.04
ALPHA_EXPERT = (SQUEEZE - SQUEEZE_LO) / (SQUEEZE_HI - SQUEEZE_LO)       # 0.45
ACTIVE = ("squeeze", "lift", "carry", "place")
DECIDE_EVERY = 2          # control steps per decision
SIDES = ("left", "right")
OBS_DIM = 10 + 6 + 2 + len(ACTIVE) + 1 + 2      # 25
ACT_DIM = 2


def alpha_of(a) -> np.ndarray:
    """Action in [-1, 1]^2 -> interpolation factor per hand around the expert's (a = 0: the expert)."""
    a = np.clip(np.asarray(a, float), -1.0, 1.0)
    return np.clip(ALPHA_EXPERT + a * np.where(a > 0, 1.0 - ALPHA_EXPERT, ALPHA_EXPERT), 0.0, 1.0)


class GripController(BimanualExpert):
    """Bimanual expert whose squeeze a policy sets while the box is held (policy None: the expert's squeeze,
    exactly its commands up to the interpolation's rounding)."""

    kind = "bimanual_grip"

    def __init__(self, env, policy=None, ik=None):
        super().__init__(env, ik)
        self.policy = policy
        self.skill_kind = "bimanual_grip" if policy is not None else "bimanual_expert"

    def reset(self, obs: dict) -> None:
        super().reset(obs)                                      # the expert's plan sets the timing
        self.plan_lo, _ = self.make_plan(obs, SQUEEZE_LO, self.timing)
        self.plan_hi, _ = self.make_plan(obs, SQUEEZE_HI, self.timing)
        assert len(self.plan_lo) == len(self.plan_hi) == len(self.plan)
        self.alpha = np.full(2, ALPHA_EXPERT)
        self.prev = np.zeros(ACT_DIM)
        self.phase_start = 0
        self.arm = {s: self.ik.arm_qadr[s] for s in SIDES}

    def command_q(self, k: int) -> np.ndarray:
        q = self.plan_lo[k][1].copy()
        for i, s in enumerate(SIDES):
            idx = self.arm[s]
            q[idx] = (1 - self.alpha[i]) * self.plan_lo[k][1][idx] + self.alpha[i] * self.plan_hi[k][1][idx]
        return q

    def observation(self, obs: dict) -> np.ndarray:
        k = min(self.k, len(self.plan) - 1)
        name = self.plan[k][0]
        q = self.command_q(k)
        lag = np.concatenate([obs["palm"][s] - self.ik.palm_pose(q, s)[0] for s in SIDES])
        start = k
        while start > 0 and self.plan[start - 1][0] == name:
            start -= 1
        end = k
        while end + 1 < len(self.plan) and self.plan[end + 1][0] == name:
            end += 1
        tote = obs["tote"]
        tilt = getattr(self.env, "tote_tilt", lambda: 0.0)()
        return np.concatenate([
            np.concatenate([obs["finger_force"][s] for s in SIDES]) / 20.0,
            np.clip(lag * 20.0, -5, 5),
            [10.0 * (tote["pos"][2] - TABLE_Z), tilt],
            [name == p for p in ACTIVE], [(k - start) / max(end - start, 1)], self.prev,
        ]).astype(np.float32)

    def set_action(self, a) -> None:
        self.prev = np.clip(np.asarray(a, float), -1.0, 1.0)
        self.alpha = alpha_of(self.prev)

    def act(self, obs: dict) -> np.ndarray:
        k = min(self.k, len(self.plan) - 1)
        name = self.plan[k][0]
        if self.policy is not None and name in ACTIVE and self.k % DECIDE_EVERY == 0:
            self.set_action(self.policy.act(self.observation(obs)))
        elif name not in ACTIVE:
            self.alpha[:] = ALPHA_EXPERT if name in ("reach", "descend") else self.alpha
        self.k += 1
        self.env.context = {"phase": f"both:{name}", "target": "tote"}
        return compose_action(self.ik, self.command_q(k), self.hand)
