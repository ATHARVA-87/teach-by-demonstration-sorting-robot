"""Node-level action semantics (review M3-1): resets and newer goals never yield a false success.

Runs ``SimNode`` in-process on its own ROS domain (no other test or running
sim can interfere) at 4x real time.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[4]
PANDA = Path(
    os.environ.get("SORTING_PANDA_XML", ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml")
)
pytestmark = pytest.mark.skipif(not PANDA.is_file(), reason="run scripts/fetch_robot_models.sh first")

READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
PARK = [2.6, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


@pytest.fixture(scope="module")
def ros() -> Iterator[Any]:
    with pytest.MonkeyPatch.context() as mp:  # environment restored afterwards
        # Isolated graph. Linux: domain IDs 102-214 put DDS ports in the ephemeral
        # port range, where discovery fails at random; stay within 1-101.
        mp.setenv("ROS_DOMAIN_ID", str(random.randint(30, 101)))
        mp.setenv("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")  # immune to stale /dev/shm segments
        mp.setenv("SORTING_PANDA_XML", str(PANDA))
        mp.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
        yield from _ros_graph()


def _ros_graph() -> Iterator[Any]:
    import rclpy
    from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor

    from sorting_sim.sim_node import SimNode

    rclpy.init(args=["--ros-args", "-p", "real_time_factor:=4.0", "-p", "scene_seed:=2"])
    sim = SimNode()
    client = rclpy.create_node("action_test_client")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(sim)
    ex.add_node(client)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    class Ros:
        node = sim
        arm = ActionClient(client, FollowJointTrajectory, "/panda_arm_controller/follow_joint_trajectory")
        hand = ActionClient(client, ParallelGripperCommand, "/panda_hand_controller/gripper_cmd")

    try:
        assert Ros.arm.wait_for_server(timeout_sec=10) and Ros.hand.wait_for_server(timeout_sec=10)
        yield Ros
    finally:
        ex.shutdown()
        sim.destroy_node()
        client.destroy_node()
        rclpy.shutdown()


def _wait(fut: Any, timeout: float = 30.0) -> Any:
    end = time.monotonic() + timeout
    while not fut.done():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)
    return fut.result()


def _send(client: Any, goal: Any) -> Any:
    handle = _wait(client.send_goal_async(goal))
    assert handle.accepted
    return handle


def _traj_goal(q: list[float], seconds: int) -> Any:
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    g = FollowJointTrajectory.Goal()
    g.trajectory.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    g.trajectory.points = [JointTrajectoryPoint(positions=q, time_from_start=Duration(sec=seconds))]
    return g


def _grip_goal(position: list[float]) -> Any:
    from control_msgs.action import ParallelGripperCommand
    from sensor_msgs.msg import JointState

    g = ParallelGripperCommand.Goal()
    g.command = JointState(name=["panda_finger_joint1"], position=position)
    return g


def _reset(node: Any, seed: int) -> None:
    from sorting_interfaces.srv import ResetScene

    resp = ResetScene.Response()
    node._on_reset(ResetScene.Request(seed=seed, n_items=4), resp)
    assert resp.success, resp.message


def test_trajectory_succeeds_and_settles(ros: Any) -> None:
    from action_msgs.msg import GoalStatus

    res = _wait(_send(ros.arm, _traj_goal(READY, 2)).get_result_async())
    assert res.status == GoalStatus.STATUS_SUCCEEDED
    assert ros.node.sim.arm_error([f"panda_joint{i}" for i in range(1, 8)], READY) < 0.005


def test_new_goal_preempts_old(ros: Any) -> None:
    from action_msgs.msg import GoalStatus

    first = _send(ros.arm, _traj_goal(PARK, 4))
    time.sleep(0.2)
    second = _send(ros.arm, _traj_goal(READY, 3))
    r1 = _wait(first.get_result_async(), timeout=5.0)  # ends at once, not after the second goal
    assert r1.status == GoalStatus.STATUS_ABORTED and "preempted" in r1.result.error_string
    assert _wait(second.get_result_async()).status == GoalStatus.STATUS_SUCCEEDED


def test_reset_aborts_trajectory(ros: Any) -> None:
    from action_msgs.msg import GoalStatus

    handle = _send(ros.arm, _traj_goal(PARK, 4))
    time.sleep(0.2)
    _reset(ros.node, 5)
    res = _wait(handle.get_result_async(), timeout=5.0)
    assert res.status == GoalStatus.STATUS_ABORTED and "scene reset" in res.result.error_string


def test_reset_during_gripper_close_is_not_a_grasp(ros: Any) -> None:
    """Reviewer's repro: used to SUCCEED with stalled=True while the hand was empty and open."""
    from action_msgs.msg import GoalStatus

    _wait(_send(ros.hand, _grip_goal([0.04])).get_result_async())
    handle = _send(ros.hand, _grip_goal([0.0]))
    _reset(ros.node, 6)
    res = _wait(handle.get_result_async(), timeout=5.0)
    assert res.status == GoalStatus.STATUS_ABORTED


def test_empty_gripper_command_rejected(ros: Any) -> None:
    from action_msgs.msg import GoalStatus

    res = _wait(_send(ros.hand, _grip_goal([])).get_result_async(), timeout=5.0)
    assert res.status == GoalStatus.STATUS_ABORTED


def test_gripper_close_on_nothing_reaches_goal(ros: Any) -> None:
    from action_msgs.msg import GoalStatus

    res = _wait(_send(ros.hand, _grip_goal([0.0])).get_result_async())
    assert res.status == GoalStatus.STATUS_SUCCEEDED
    assert res.result.reached_goal and not res.result.stalled


def _raw_traj_goal(names: list[str], points: list[tuple[list[float], float]]) -> Any:
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    g = FollowJointTrajectory.Goal()
    g.trajectory.joint_names = names
    for q, t in points:
        sec = int(t)
        g.trajectory.points.append(
            JointTrajectoryPoint(positions=q, time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9)))
        )
    return g


@pytest.mark.parametrize(
    ("names", "points"),
    [
        (["panda_joint1"], [([float("nan")], 1.0)]),  # used to come back SUCCEEDED
        ([], [([], 0.0), ([], 1.0)]),  # used to freeze physics and /clock
    ],
    ids=["nan", "no_joints"],
)
def test_malformed_goal_rejected_and_physics_keeps_running(ros: Any, names: list[str], points: Any) -> None:
    from action_msgs.msg import GoalStatus
    from control_msgs.action import FollowJointTrajectory

    res = _wait(_send(ros.arm, _raw_traj_goal(names, points)).get_result_async(), timeout=5.0)
    assert res.status == GoalStatus.STATUS_ABORTED
    assert res.result.error_code == FollowJointTrajectory.Result.INVALID_JOINTS
    t0 = ros.node.sim.time
    time.sleep(0.3)
    assert ros.node.sim.time > t0  # sim time (and /clock) still advancing


def test_nan_gripper_goal_rejected(ros: Any) -> None:
    from action_msgs.msg import GoalStatus

    res = _wait(_send(ros.hand, _grip_goal([float("nan")])).get_result_async(), timeout=5.0)
    assert res.status == GoalStatus.STATUS_ABORTED
