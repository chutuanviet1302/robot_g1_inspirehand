"""Plain-Python conversions between HomeHand data and ROS 2 message fields (no rclpy import: unit-testable
anywhere, the nodes only copy these values into messages).

Topics (see README "ROS 2"):
  /joint_states                 sensor_msgs/JointState     26 actuated joints (14 arm + 12 hand), measured
  /homehand/observation         std_msgs/String            JSON: step, palms, finger forces, perceived objects
  /joint_command                trajectory_msgs/JointTrajectory  one point = the 26 joint targets; header.frame_id
                                                           = command_tag(): the step it answers (latency, lockstep)
                                                           and whether the controller has finished
  /inspire/{left,right}/angle_cmd  std_msgs/Int32MultiArray  RH56DFX angle registers 0..1000 (1000 = open)
  /inspire/{left,right}/force_limit std_msgs/Int32MultiArray RH56DFX force-limit registers
"""

from __future__ import annotations

import json

import numpy as np

from homehand.model import spec

JOINT_NAMES = list(spec.ACTUATORS)
N_JOINTS = len(JOINT_NAMES)     # 26, the order of the simulation action


def joint_state(obs: dict) -> tuple[list[str], list[float]]:
    """Measured joint positions in action order (arm 14, left hand 6, right hand 6)."""
    q = np.concatenate([obs["arm_q"], obs["hand_q"]]).astype(float)
    return JOINT_NAMES, q.tolist()


def _encode(x):
    if isinstance(x, dict):
        return {str(k): _encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_encode(v) for v in x]
    if isinstance(x, np.ndarray):
        return {"__nd__": x.tolist()}
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def _decode(x):
    if isinstance(x, dict):
        if set(x) == {"__nd__"}:
            return np.asarray(x["__nd__"], float)
        return {k: _decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v) for v in x]
    return x


def obs_to_json(obs: dict) -> str:
    """The observation the controllers use (arrays kept as arrays). Exact float round trip: JSON prints the
    shortest repr that parses back to the same double."""
    return json.dumps(_encode(obs))


def obs_from_json(s: str) -> dict:
    return _decode(json.loads(s))


def command_positions(action26) -> list[float]:
    a = np.asarray(action26, float)
    if a.shape != (N_JOINTS,):
        raise ValueError(f"expected {N_JOINTS} joint targets, got shape {a.shape}")
    return a.tolist()


def action_from_command(names: list[str], positions: list[float]) -> np.ndarray | None:
    """26-dim action from a JointTrajectory point (any joint order); None if a joint is missing or not finite."""
    idx = {n: i for i, n in enumerate(names)}
    if any(n not in idx for n in JOINT_NAMES) or len(positions) != len(names):
        return None
    a = np.array([positions[idx[n]] for n in JOINT_NAMES], float)
    return a if np.all(np.isfinite(a)) else None


def command_tag(step: int, done: bool) -> str:
    return f"{int(step)}:done" if done else str(int(step))


def parse_tag(tag: str) -> tuple[int, bool]:
    step, _, flag = tag.partition(":")
    return (int(step) if step.isdigit() else -1), flag == "done"
