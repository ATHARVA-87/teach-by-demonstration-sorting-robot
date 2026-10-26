"""Teach-only mode: AR teaching (webcam or scripted hand) + the rule learner.

The learner listens to demonstrations, asks questions back through the AR
view, and writes the task file when ``/rule_learner/confirm`` is called.

ros2 launch sorting_bringup teach.launch.py
ros2 launch sorting_bringup teach.launch.py source:=synthetic   # no webcam needed
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ar_params = os.path.join(get_package_share_directory("ar_teach"), "config", "ar_teach.yaml")
    args = [
        DeclareLaunchArgument("source", default_value="webcam", description="webcam | synthetic"),
        DeclareLaunchArgument("scene_seed", default_value="0", description="Random scene to teach on"),
        DeclareLaunchArgument("scene_config", default_value="", description="scene.yaml (empty: installed)"),
        DeclareLaunchArgument(
            "tasks_dir", default_value="tasks", description="Where confirmed task files go"
        ),
    ]
    teach = Node(
        package="ar_teach",
        executable="teach_node",
        name="ar_teach",
        output="screen",
        parameters=[
            ar_params,
            {
                "source": LaunchConfiguration("source"),
                "scene_seed": LaunchConfiguration("scene_seed"),
                "scene_config": LaunchConfiguration("scene_config"),
            },
        ],
    )
    learner = Node(
        package="rule_learner",
        executable="rule_learner_node",
        name="rule_learner",
        output="screen",
        parameters=[
            os.path.join(get_package_share_directory("rule_learner"), "config", "rule_learner.yaml"),
            {
                "scene_config": LaunchConfiguration("scene_config"),
                "tasks_dir": LaunchConfiguration("tasks_dir"),
            },
        ],
    )
    return LaunchDescription([*args, teach, learner])
