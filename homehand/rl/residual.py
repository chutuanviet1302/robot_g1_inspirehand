"""Residual policy: a small learned correction added to the expert's joint targets while the hand is in contact.

The scripted expert (`control.skills.PickPlaceSkill`) plans collision-free reach / carry / return motions well;
what is left are contact problems (a grasp that slips, fingers nudging the object or a neighbour, an object
lost while lifting, more so with low friction). The residual acts only in the contact phases
(`ACTIVE_PHASES`), adds at most `ARM_SCALE` / `HAND_SCALE` to the 13 joint targets of the acting side, and
everything still goes through the safety layer.

Observation (realistic sensors only): the imitation-learning features (`policy.features.skill_features`:
proprioception, perceived object, grasp plan), the phase, the per-finger contact forces (the real Inspire hand
reports one force per finger) and the previous residual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from homehand.control.skills import PickPlaceSkill
from homehand.policy.features import OBS_DIM as SKILL_OBS_DIM
from homehand.policy.features import side_slices, skill_features

ACTIVE_PHASES = ("approach", "grasp", "lift")
ARM_SCALE = 0.04     # rad: largest correction of an arm joint target
HAND_SCALE = 0.15    # rad: largest correction of a hand joint target
FORCE_SCALE = 20.0   # N: finger forces are divided by this
ACT_DIM = 13
OBS_DIM = SKILL_OBS_DIM + len(ACTIVE_PHASES) + 5 + ACT_DIM    # 39 + 3 + 5 + 13 = 60
SCALE = np.r_[np.full(7, ARM_SCALE), np.full(6, HAND_SCALE)].astype(np.float32)


def residual_obs(obs: dict, skill: PickPlaceSkill, prev: np.ndarray) -> np.ndarray:
    phase = np.array([skill.phase == p for p in ACTIVE_PHASES], np.float32)
    force = np.asarray(obs["finger_force"][skill.side], np.float32) / FORCE_SCALE
    return np.concatenate([skill_features(obs, skill.side, skill.target, skill.steps), phase,
                           np.clip(force, 0.0, 5.0), prev]).astype(np.float32)


def apply_residual(base26: np.ndarray, side: str, a: np.ndarray) -> np.ndarray:
    """Expert action + scaled residual (a in [-1, 1]^13) on the acting side's joint targets."""
    a = np.clip(np.asarray(a, float), -1.0, 1.0) * SCALE
    out = np.array(base26, dtype=float)
    arm, hand = side_slices(side)
    out[:14][arm] += a[:7]
    out[14:][hand] += a[7:]
    return out


class ResidualPolicy:
    """A trained residual (SB3 PPO actor) with its observation normalisation, for inference."""

    def __init__(self, path: Path | str, device: str = "cpu"):
        from stable_baselines3 import PPO
        path = Path(path)
        self.model = PPO.load(path / "policy.zip", device=device)
        stats = np.load(path / "obs_stats.npz")
        self.mean, self.var, self.clip = stats["mean"], stats["var"], float(stats["clip"])
        self.kind = "residual"

    def act(self, x: np.ndarray) -> np.ndarray:
        x = np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)
        a, _ = self.model.predict(x.astype(np.float32), deterministic=True)
        return a


class ResidualSkill:
    """`PickPlaceSkill` + residual in the contact phases; same interface, so the planner uses it like the expert.
    With `policy=None` (or a zero residual) it behaves exactly like the expert."""

    def __init__(self, policy: ResidualPolicy | None, *args, **kwargs):
        self.expert = PickPlaceSkill(*args, **kwargs)
        self.policy = policy
        self.prev = np.zeros(ACT_DIM, np.float32)

    def __getattr__(self, name):              # q, hand_cmd, status, phase, done, target, side, steps, clearance
        return getattr(self.expert, name)

    def act(self, obs: dict) -> np.ndarray:
        active = self.expert.phase in ACTIVE_PHASES
        x = residual_obs(obs, self.expert, self.prev) if active and self.policy is not None else None
        base = self.expert.act(obs)
        if x is None:
            self.prev[:] = 0.0
            return base
        a = self.policy.act(x)
        self.prev = np.asarray(a, np.float32)
        return apply_residual(base, self.expert.side, a)
