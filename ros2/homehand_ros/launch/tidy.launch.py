"""Simulation bridge + controller + Inspire bridge for one episode.

    ros2 launch homehand_ros tidy.launch.py seed:=3 controller:=expert randomization:=medium
    ros2 launch homehand_ros tidy.launch.py task:=bimanual_lift
    ros2 launch homehand_ros tidy.launch.py lockstep:=false      # real-time stepping at 25 Hz
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arg = LaunchConfiguration
    args = [DeclareLaunchArgument("seed", default_value="0"),
            DeclareLaunchArgument("task", default_value="tidy_table"),
            DeclareLaunchArgument("randomization", default_value="nominal"),
            DeclareLaunchArgument("controller", default_value="expert"),
            DeclareLaunchArgument("lockstep", default_value="true")]
    sim = Node(package="homehand_ros", executable="sim_bridge", name="homehand_sim", output="screen",
               parameters=[{"seed": ParameterValue(arg("seed"), value_type=int), "task": arg("task"),
                            "randomization": arg("randomization"),
                            "lockstep": ParameterValue(arg("lockstep"), value_type=bool)}])
    ctrl = Node(package="homehand_ros", executable="tidy_controller", name="homehand_controller", output="screen",
                parameters=[{"controller": arg("controller"), "task": arg("task")}])
    hand = Node(package="homehand_ros", executable="inspire_bridge", name="homehand_inspire_bridge", output="screen")
    # the episode ends when the simulation node exits: stop everything
    stop = RegisterEventHandler(OnProcessExit(target_action=sim, on_exit=[Shutdown()]))
    return LaunchDescription([*args, sim, ctrl, hand, stop])
