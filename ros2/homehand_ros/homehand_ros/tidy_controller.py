"""ROS 2 controller node: the HomeHand planner + skill (expert, residual RL, ACT / Diffusion / flow) driven only by
topics. It never touches the simulator: what it knows comes from /homehand/observation, what it does goes out on
/joint_command - the same interface a real-robot bridge would offer.

Parameters
  controller ("expert")   expert | residual_* (models/<name>) | a model in models/ (ACT, Diffusion, flow)
  task ("tidy_table")     tidy_table | pick_place | bimanual_lift (the bimanual expert)

Perception: the observation carries the object poses (ground truth from the simulator, the planner's
"ground_truth" mode); a camera-image topic + the YOLO detector would replace it on hardware.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from homehand_ros.controllers import make_controller
from homehand_ros.convert import JOINT_NAMES, command_positions, command_tag, obs_from_json


class TidyController(Node):
    def __init__(self):
        super().__init__("homehand_controller")
        self.declare_parameter("controller", "expert")
        self.declare_parameter("task", "tidy_table")
        c, task = self.get_parameter("controller").value, self.get_parameter("task").value
        self.ctrl = make_controller(c, task)
        self.started = False
        self.pub = self.create_publisher(JointTrajectory, "/joint_command", 10)
        self.create_subscription(String, "/homehand/observation", self.on_obs, 10)
        self.get_logger().info(f"controller {c} for {task}")

    def on_obs(self, msg: String) -> None:
        obs = obs_from_json(msg.data)
        if not self.started:
            self.ctrl.reset(obs)
            self.started = True
        a = self.ctrl.act(obs)
        out = JointTrajectory(joint_names=JOINT_NAMES,
                              points=[JointTrajectoryPoint(positions=command_positions(a))])
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = command_tag(int(obs["t"]), bool(self.ctrl.done))
        self.pub.publish(out)


def main():
    rclpy.init()
    node = TidyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
