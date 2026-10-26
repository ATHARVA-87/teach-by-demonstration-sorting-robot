"""Launch files load and describe the expected nodes (no processes started)."""

import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node

LAUNCH_DIR = Path(__file__).resolve().parent.parent / "launch"


def _load(name: str) -> LaunchDescription:
    spec = importlib.util.spec_from_file_location(name, LAUNCH_DIR / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def test_teach_launch_describes_ar_teach_node() -> None:
    ld = _load("teach.launch.py")
    nodes = [e for e in ld.entities if isinstance(e, Node)]
    assert len(nodes) == 2  # ar_teach + rule_learner
    args = {e.name for e in ld.entities if type(e).__name__ == "DeclareLaunchArgument"}
    assert {"source", "scene_seed", "scene_config", "tasks_dir"} <= args
