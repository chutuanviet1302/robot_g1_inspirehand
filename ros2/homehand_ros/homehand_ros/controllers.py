"""Controllers for the ROS 2 controller node (no rclpy import, so the episode logic is testable without ROS)."""

from __future__ import annotations

import json
from types import SimpleNamespace


def make_controller(controller: str, task: str):
    """The controller and a stand-in for the env with the two things the planner takes from it (`context` for
    failure attribution, `obj_info` for object heights)."""
    from homehand import paths
    stub = SimpleNamespace(context={}, obj_info=json.loads(paths.OBJECTS_JSON.read_text()))
    if task == "bimanual_lift":
        from homehand.control.bimanual import BimanualExpert
        return BimanualExpert(stub)
    from homehand.control.planner import TidyPlanner
    policy = None
    if controller.startswith("residual"):
        from homehand.rl.residual import ResidualPolicy
        policy = ResidualPolicy(paths.MODELS_DIR / controller)
    elif controller != "expert":
        from homehand.policy.skill import load_policy
        policy = load_policy(controller, device="cpu")
    return TidyPlanner(stub, perception="ground_truth", policy=policy)
