"""Feature vector of the pick-and-place skill, shared by data collection and policy inference.

The policy controls one arm + hand at a time (13 actions); which side it is, which object (grasp type) and
where it is (from perception) are part of the input, so a single network serves both hands.
"""

from __future__ import annotations

import numpy as np

from homehand.control.grasps import GRASPS
from homehand.model import spec

GRASP_NAMES = list(GRASPS)
PROGRESS_HORIZON = 450.0   # control steps (a pick-and-place skill takes ~250-400); progress saturates at 1
FEATURE_NAMES = (
    [f"arm_q{i}" for i in range(7)] + [f"hand_q{i}" for i in range(6)] + ["palm_x", "palm_y", "palm_z"]
    + ["obj_x", "obj_y", "obj_h", "lying", "yaw_sin", "yaw_cos"]
    + ["grasp_x", "grasp_y", "grasp_z", "drop_x", "drop_y", "drop_z"]
    + [f"grasp_{g}" for g in GRASP_NAMES] + ["side_left", "side_right", "progress"]
)
OBS_DIM = len(FEATURE_NAMES)        # 39
ACT_DIM = spec.N_ARM + spec.N_HAND  # 13


def side_slices(side: str) -> tuple[slice, slice]:
    """(arm, hand) slices of the 26-dim action / of obs['arm_q'] (14) and obs['hand_q'] (12)."""
    i = 0 if side == "left" else 1
    return slice(7 * i, 7 * i + 7), slice(6 * i, 6 * i + 6)


def skill_features(obs: dict, side: str, target, step: int) -> np.ndarray:
    """Proprioception + the perceived object + the grasp plan derived from it (palm pose at the grasp and over
    the bin, which grasp type, wrist yaw) + side + progress. The plan is computed from perception only, so the
    policy still never sees simulator state."""
    from homehand.control.skills import grasp_plan
    arm, hand = side_slices(side)
    p = grasp_plan(side, target)
    onehot = np.zeros(len(GRASP_NAMES))
    onehot[GRASP_NAMES.index(p["grasp_type"].name)] = 1.0
    return np.concatenate([
        obs["arm_q"][arm], obs["hand_q"][hand], obs["palm"][side],
        [target.pos[0], target.pos[1], target.height, float(target.rest == "lying"),
         np.sin(p["hand_yaw"]), np.cos(p["hand_yaw"])],
        p["grasp"], p["drop"], onehot, [side == "left", side == "right"], [min(1.0, step / PROGRESS_HORIZON)],
    ]).astype(np.float32)


def side_action(action26: np.ndarray, side: str) -> np.ndarray:
    arm, hand = side_slices(side)
    return np.concatenate([action26[:14][arm], action26[14:][hand]]).astype(np.float32)


def merge_action(base26: np.ndarray, side: str, a13: np.ndarray) -> np.ndarray:
    out = np.array(base26, dtype=float)
    arm, hand = side_slices(side)
    out[:14][arm] = a13[:7]
    out[14:][hand] = a13[7:]
    return out
