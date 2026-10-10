"""Run a fine-tuned SmolVLA checkpoint as the pick-and-place skill (LeRobot environment, see kaggle/README.md).

The task planner stays the high level: it picks the object and the hand and hands the policy an instruction such
as "pick up the mustard bottle with the left hand and put it in the bin". SmolVLA sees the three camera images
(head, both wrists, 256x256), the 26 joint positions and the instruction, and outputs chunks of 50 joint targets
(LeRobot's action queue executes them). Only the active arm and hand follow the policy; the other side holds
its pose. Every command still goes through the safety layer of the environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from homehand.control.grasps import RELAXED_HAND
from homehand.control.skills import compose_action
from homehand.data.record_vla import instruction, render_cameras, state_vector
from homehand.policy.features import merge_action, side_action

# SmolVLA (smolvla_base) names its three image inputs camera1..3: head, left wrist, right wrist
IMAGE_KEYS = {"head": "observation.images.camera1", "L_wrist_cam": "observation.images.camera2",
              "R_wrist_cam": "observation.images.camera3"}


class SmolVLARunner:
    kind = "smolvla"

    def __init__(self, path: str | Path, device: str | None = None, held_out_language: bool = False,
                 max_steps: int = 520, min_steps: int = 150):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        path = Path(path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = SmolVLAPolicy.from_pretrained(str(path)).to(self.device).eval()
        self.policy.config.device = str(self.device)
        # the processors remember the training device (cpu on a smoke run, cuda on Kaggle): use ours
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        self.held_out = held_out_language
        self.max_steps, self.min_steps = max_steps, min_steps
        self.name = path.name

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: dict, images: dict, task: str) -> np.ndarray:
        from lerobot.common.control_utils import predict_action
        o = {"observation.state": state_vector(obs)}
        for cam, key in IMAGE_KEYS.items():
            o[key] = images[cam]
        a = predict_action(o, self.policy, self.device, self.pre, self.post, use_amp=False, task=task,
                           robot_type="unitree_g1_inspire")
        return np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, float).reshape(-1)


class SmolVLASkill:
    """Same interface as `PickPlaceSkill` / `PolicySkill`."""

    HOME_TOL = 0.06
    HOME_STEPS = 8

    def __init__(self, runner: SmolVLARunner, env, ik, side: str, target, q_start: np.ndarray, hand_cmd: dict):
        self.runner, self.env, self.ik, self.side, self.target = runner, env, ik, side, target
        self.q = q_start.copy()
        self.hand_cmd = {s: np.asarray(v, float).copy() for s, v in hand_cmd.items()}
        self.base = compose_action(ik, self.q, self.hand_cmd)
        self.task = instruction(target.name, side, k=0, held_out=runner.held_out)
        self.status = "policy"
        self.steps = self.home_count = 0
        arm = slice(0, 7) if side == "left" else slice(7, 14)
        self.home_arm = ik.arm_targets(ik.q_home)[arm]
        runner.reset()

    @property
    def done(self) -> bool:
        return self.steps >= self.runner.max_steps or self.home_count >= self.HOME_STEPS

    @property
    def phase(self) -> str:
        return f"smolvla {self.steps}/{self.runner.max_steps}"

    def act(self, obs: dict) -> np.ndarray:
        a26 = self.runner.act(obs, render_cameras(self.env), self.task)
        a13 = side_action(a26, self.side)
        self.steps += 1
        at_home = self.steps >= self.runner.min_steps and np.max(np.abs(a13[:7] - self.home_arm)) < self.HOME_TOL
        self.home_count = self.home_count + 1 if at_home else 0
        a = merge_action(self.base, self.side, a13)
        if self.done:
            self.q = self.ik.q_home.copy()
            self.hand_cmd[self.side] = np.array(RELAXED_HAND, float)
        return a
