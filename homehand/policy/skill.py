"""Run a learned policy (ACT / Diffusion Policy) as the pick-and-place skill inside the task planner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from homehand import paths
from homehand.control.ik import ArmIK
from homehand.control.skills import Target, compose_action
from homehand.policy.features import merge_action, side_action, skill_features
from homehand.control.grasps import RELAXED_HAND
from homehand.policy.models import MODELS

DIFFUSION_INFER_STEPS = 30


class LoadedPolicy:
    """A trained checkpoint + normalisation + execution strategy.

    ACT: query every step, temporal ensembling of overlapping chunks (weights exp(-m * age)).
    Diffusion: receding horizon - execute `n_exec` actions of each sampled chunk, then re-plan.
    """

    def __init__(self, path: Path | str, device: str | None = None, n_exec: int = 8, ensemble_m: float = 0.05):
        path = Path(path)
        if path.is_dir():
            path = path / "policy.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.kind = ck["kind"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MODELS[self.kind](**ck["cfg"]).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"])
        if self.kind == "diffusion":
            # 10 DDIM steps (the training default) doubled the action error of our checkpoint vs. 30 steps
            # (validation action-MSE 0.157 -> 0.075; 100 steps: no further gain)
            self.model.n_infer = DIFFUSION_INFER_STEPS
        self.stats = {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}
        self.max_steps = int(ck["demo_length"]["max"]) + 15
        self.min_steps = int(0.6 * ck["demo_length"].get("mean", ck["demo_length"]["max"]))
        self.n_exec, self.m = n_exec, ensemble_m
        self.name = path.parent.name
        self.gen = torch.Generator(device=self.device)

    def reset(self, seed: int = 0) -> None:
        self.t = 0
        self.history: list[tuple[int, np.ndarray]] = []   # (start step, chunk) for ACT ensembling
        self.plan: list[np.ndarray] = []                  # pending actions for diffusion
        self.gen.manual_seed(seed)

    def _infer(self, feats: np.ndarray) -> np.ndarray:
        x = torch.from_numpy((feats - self.stats["obs_mean"]) / self.stats["obs_std"]).float()[None].to(self.device)
        if self.kind == "diffusion":
            y = self.model.predict(x, generator=self.gen)[0].cpu().numpy()
        else:
            y = self.model.predict(x)[0].cpu().numpy()
        return y * self.stats["act_std"] + self.stats["act_mean"]

    def act(self, feats: np.ndarray) -> np.ndarray:
        if self.kind == "diffusion":
            if not self.plan:
                self.plan = list(self._infer(feats)[: self.n_exec])
            a = self.plan.pop(0)
        else:
            self.history.append((self.t, self._infer(feats)))
            self.history = [(s, c) for s, c in self.history if self.t - s < len(c)]
            preds = np.stack([c[self.t - s] for s, c in self.history])
            ages = np.array([self.t - s for s, _ in self.history], dtype=float)
            w = np.exp(-self.m * ages)
            a = (w[:, None] * preds).sum(0) / w.sum()
        self.t += 1
        return a


def load_policy(name_or_path: str, device: str | None = None) -> LoadedPolicy:
    p = Path(name_or_path)
    if not p.exists():
        p = paths.MODELS_DIR / name_or_path
    return LoadedPolicy(p, device=device)


class PolicySkill:
    """Same interface as `PickPlaceSkill`, driven by a learned policy for a fixed horizon."""

    def __init__(self, policy: LoadedPolicy, ik: ArmIK, side: str, target: Target, q_start: np.ndarray,
                 hand_cmd: dict):
        self.policy, self.ik, self.side, self.target = policy, ik, side, target
        self.q = q_start.copy()
        self.hand_cmd = {s: np.asarray(v, float).copy() for s, v in hand_cmd.items()}
        self.base = compose_action(ik, self.q, self.hand_cmd)
        self.status = "policy"
        self.steps = 0
        self.home_count = 0
        arm = slice(0, 7) if side == "left" else slice(7, 14)
        self.home_arm = ik.arm_targets(ik.q_home)[arm]
        policy.reset(seed=hash((target.name, side)) % 2**31)

    HOME_TOL = 0.06          # rad: the commanded arm is back in the ready pose
    HOME_STEPS = 8           # ... for this many consecutive steps

    @property
    def done(self) -> bool:
        # The policy has no "finished" output: the skill ends once the policy has brought the arm back to the
        # ready pose and holds it there (proprioceptive, like the expert's last phase), or at the horizon.
        # (A fixed horizon of max demo length made every pick last 532 steps: four of them overran the episode.)
        return self.steps >= self.policy.max_steps or self.home_count >= self.HOME_STEPS

    @property
    def phase(self) -> str:
        return f"{self.policy.kind} {self.steps}/{self.policy.max_steps}"

    def act(self, obs: dict) -> np.ndarray:
        feats = skill_features(obs, self.side, self.target, self.steps)
        a13 = self.policy.act(feats)
        self.steps += 1
        at_home = self.steps >= self.policy.min_steps and np.max(np.abs(a13[:7] - self.home_arm)) < self.HOME_TOL
        self.home_count = self.home_count + 1 if at_home else 0
        a = merge_action(self.base, self.side, a13)
        if self.done:
            # hand back the arm in a known state: the planner continues from the ready pose
            self.q = self.ik.q_home.copy()
            self.hand_cmd[self.side] = np.array(RELAXED_HAND, float)
        return a
