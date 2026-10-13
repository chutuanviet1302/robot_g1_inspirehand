"""ROS 2 node: the hand part of /joint_command in the Inspire RH56DFX driver's format, one stream per hand.

Uses `homehand.control.inspire_real.InspireBridge` (joint radians -> six normalised targets, 1 = open, channel
order pinky .. thumb rotation, rate-limited, non-finite commands held) and publishes the angle registers
(0..1000) plus the per-finger force limit (<= the rated 9.8 N) - what a driver such as Unitree's
dfx_inspire_service writes to the hand over Modbus. With no hand connected the topics can be recorded with
`ros2 bag record` and checked offline.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from trajectory_msgs.msg import JointTrajectory

from homehand.control.inspire_real import InspireBridge, split_action
from homehand_ros.convert import action_from_command


class InspireBridgeNode(Node):
    def __init__(self):
        super().__init__("homehand_inspire_bridge")
        self.declare_parameter("force_limit_n", 6.0)
        f = float(self.get_parameter("force_limit_n").value)
        self.bridges = {s: InspireBridge(side=s, force_limit_n=f) for s in ("left", "right")}
        self.pub_angle = {s: self.create_publisher(Int32MultiArray, f"/inspire/{s}/angle_cmd", 10) for s in self.bridges}
        self.pub_force = {s: self.create_publisher(Int32MultiArray, f"/inspire/{s}/force_limit", 10) for s in self.bridges}
        self.create_subscription(JointTrajectory, "/joint_command", self.on_command, 10)
        self.create_timer(1.0, self.publish_force)

    def publish_force(self) -> None:
        for s, b in self.bridges.items():
            self.pub_force[s].publish(Int32MultiArray(data=[int(v) for v in b.force_registers()]))

    def on_command(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        a = action_from_command(list(msg.joint_names), list(msg.points[0].positions))
        hands = split_action(a) if a is not None else {s: np.full(6, np.nan) for s in self.bridges}
        for s, b in self.bridges.items():           # a bad command is held by the bridge, never forwarded
            self.pub_angle[s].publish(Int32MultiArray(data=[int(v) for v in b.to_registers(hands[s])]))


def main():
    rclpy.init()
    node = InspireBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():          # Ctrl-C / launch shutdown already closed the context
            rclpy.shutdown()
