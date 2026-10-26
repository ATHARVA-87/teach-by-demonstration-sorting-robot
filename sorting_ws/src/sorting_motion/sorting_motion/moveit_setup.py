"""Build the MoveIt configuration dictionary for moveit_py.

Robot model, SRDF, kinematics and Pilz limits come from MoveIt's
``moveit_resources_panda`` packages; the controllers (our action contract) and
moveit_py settings come from this package.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import JointState

PACKAGE = "sorting_motion"


def config_dir() -> Path:
    """Installed ``share/sorting_motion/config`` (falls back to the source tree)."""
    try:
        return Path(get_package_share_directory(PACKAGE)) / "config"
    except PackageNotFoundError:  # not installed: running from the source tree
        return Path(__file__).resolve().parents[1] / "config"


def moveit_config(use_sim_time: bool = True, offline: bool = False) -> dict[str, Any]:
    """Parameters for ``MoveItPy(config_dict=...)``.

    Args:
        use_sim_time: follow ``/clock`` (needed with the simulator so that
            execution-duration monitoring uses sim time).
        offline: plan without a running robot (no joint states to wait for).
    """
    cfg_dir = config_dir()
    cfg = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .planning_pipelines(pipelines=["pilz_industrial_motion_planner", "ompl"])
        .pilz_cartesian_limits()
        .trajectory_execution(file_path=str(cfg_dir / "moveit_controllers.yaml"))
        .moveit_cpp(file_path=str(cfg_dir / "moveit_py.yaml"))
        .to_moveit_configs()
    ).to_dict()
    cfg["use_sim_time"] = use_sim_time
    if offline:
        cfg["planning_scene_monitor_options"]["wait_for_initial_state_timeout"] = 0.0
    return cfg


ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))

# MoveIt's 'ready' pose turned to face backwards: keeps the arm out of the overhead
# camera (same as sorting_sim.cell.PARK_POSE; not imported, so the motion side
# doesn't depend on the simulator package).
PARK_POSE = (2.6, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
READY_POSE = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)  # MoveIt's named state 'ready'

# Panda arm joint limits (rad), as in the URDF MoveIt uses.
ARM_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
ARM_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)


def within_limits(q: Any, margin: float = 0.0) -> bool:
    """True if every arm joint is at least ``margin`` inside its limits."""
    q = np.asarray(q, float)
    return bool(np.all(q >= np.asarray(ARM_LOWER) + margin) and np.all(q <= np.asarray(ARM_UPPER) - margin))


def wait_for_state_sync(moveit: Any, node: Node, timeout: float = 10.0, tol: float = 0.01) -> bool:
    """Wait until MoveIt's current state matches the robot's ``/joint_states``.

    ``MoveItPy`` returns before its state monitor has received a joint state;
    planning right away starts from the all-zero default pose, which is in
    self-collision for the Panda ("panda_hand - panda_link5"). The C++
    ``wait_for_current_robot_state`` binding needs an ``rclcpp::Time``, which
    Python can't build, hence this check.

    Returns:
        True once in sync, False on timeout.
    """
    end = time.monotonic() + timeout
    psm = moveit.get_planning_scene_monitor()
    while time.monotonic() < end:
        # A fresh message every round: one taken before a scene reset would be stale
        # forever (seen: sync "failed" after a reset that moved the arm).
        ok, js = wait_for_message(JointState, node, "/joint_states", time_to_wait=1.0)
        if not ok or not isinstance(js, JointState):
            continue
        want = np.array([js.position[js.name.index(j)] for j in ARM_JOINTS])
        with psm.read_only() as scene:
            have = np.asarray(scene.current_state.get_joint_group_positions("panda_arm"))
        if np.max(np.abs(have - want)) < tol:
            return True
        time.sleep(0.05)
    return False
