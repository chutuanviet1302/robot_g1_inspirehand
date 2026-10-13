"""ROS 2 node wrapping the MuJoCo simulation: publishes joint states + the observation, applies /joint_command.

Parameters
  seed, randomization, task      the episode (tidy_table | pick_place | bimanual_lift)
  lockstep (true)                step only once the command answering the last observation arrived (timeout
                                 `command_timeout` s, then the last command is held): the episode is then exactly
                                 the one `homehand` runs offline, whatever the transport latency
  rate (25.0)                    control rate in Hz when lockstep is false (real-time stepping)

The safety layer stays inside the simulation, as it sits in front of the motors on the robot. At the end of the
episode the outcome and the measured command latency are published on /homehand/episode (JSON) and logged.
"""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from homehand_ros.convert import action_from_command, joint_state, obs_to_json, parse_tag


class SimBridge(Node):
    def __init__(self):
        super().__init__("homehand_sim")
        self.declare_parameter("seed", 0)
        self.declare_parameter("randomization", "nominal")
        self.declare_parameter("task", "tidy_table")
        self.declare_parameter("lockstep", True)
        self.declare_parameter("rate", 25.0)
        self.declare_parameter("command_timeout", 2.0)
        self.declare_parameter("settle_steps", 15)
        p = lambda n: self.get_parameter(n).value
        self.task, self.lockstep, self.timeout = p("task"), p("lockstep"), p("command_timeout")
        if self.task == "bimanual_lift":
            from homehand.env.bimanual_env import BimanualEnv
            self.env = BimanualEnv()
        else:
            from homehand.env.kitchen_env import KitchenEnv
            self.env = KitchenEnv()
        self.obs = self.env.reset(self.task, seed=p("seed"), randomization=p("randomization"))
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_obs = self.create_publisher(String, "/homehand/observation", 10)
        self.pub_ep = self.create_publisher(String, "/homehand/episode", 10)
        self.create_subscription(JointTrajectory, "/joint_command", self.on_command, 10)
        self.cmd, self.cmd_for, self.sent_at = None, -1, {}
        self.latency, self.done, self.controller_done, self.settle = [], False, False, p("settle_steps")
        self.publish()
        period = 0.002 if self.lockstep else 1.0 / p("rate")
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(f"{self.task} seed {p('seed')} ({p('randomization')}), lockstep={self.lockstep}")

    def publish(self) -> None:
        names, q = joint_state(self.obs)
        js = JointState(name=names, position=q)
        js.header.stamp = self.get_clock().now().to_msg()
        self.pub_js.publish(js)
        self.pub_obs.publish(String(data=obs_to_json(self.obs)))
        self.sent_at[int(self.obs["t"])] = time.perf_counter()

    def on_command(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        a = action_from_command(list(msg.joint_names), list(msg.points[0].positions))
        if a is None:
            self.get_logger().warn("ignored a command with missing / non-finite joint targets")
            return
        t, done = parse_tag(msg.header.frame_id)
        if t in self.sent_at:
            self.latency.append(time.perf_counter() - self.sent_at.pop(t))
        self.cmd, self.cmd_for = a, t
        self.controller_done |= done

    def tick(self) -> None:
        if self.done:
            return
        t = int(self.obs["t"])
        if self.lockstep and self.cmd_for != t:
            if time.perf_counter() - self.sent_at.get(t, time.perf_counter()) < self.timeout:
                return                       # wait for the answer to this observation
            self.get_logger().warn(f"no command for step {t}: holding the last one")
        action = self.cmd if self.cmd is not None else self.env.last_action
        self.obs, done = self.env.step(action)
        if self.controller_done and not done:
            self.settle -= 1
            if self.settle <= 0:
                self.env.force_finish()
                done = True
        if done:
            self.finish()
            return
        self.publish()

    def finish(self) -> None:
        self.done = True
        info = self.env.info
        lat = np.array(self.latency) * 1e3 if self.latency else np.zeros(1)
        ep = {"task": info.task, "seed": info.seed, "outcome": info.outcome, "object_outcomes": info.object_outcomes,
              "steps": info.steps, "latency_ms": {"mean": round(float(lat.mean()), 2),
                                                 "p95": round(float(np.percentile(lat, 95)), 2),
                                                 "max": round(float(lat.max()), 2), "n": int(len(self.latency))}}
        self.pub_ep.publish(String(data=json.dumps(ep, default=str)))
        self.get_logger().info(f"episode finished: {json.dumps(ep, default=str)}")


def main():
    rclpy.init()
    node = SimBridge()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(10):                  # let the episode message go out
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if rclpy.ok():          # Ctrl-C / launch shutdown already closed the context
            rclpy.shutdown()
