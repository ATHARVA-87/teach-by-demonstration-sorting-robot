"""Tests for the simulation core (MuJoCo only, no ROS)."""

import os
from pathlib import Path

import numpy as np
import pytest

from sorting_scene import Item, Scene, generate_scene, load_config
from sorting_sim.cell import PARK_POSE
from sorting_sim.sim_core import FINGER_MAX, SimCore

ROOT = Path(__file__).resolve().parents[4]
PANDA = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
SCENE_YAML = ROOT / "sorting_ws" / "src" / "sorting_scene" / "config" / "scene.yaml"
pytestmark = pytest.mark.skipif(not PANDA.is_file(), reason="run scripts/fetch_robot_models.sh first")


@pytest.fixture()
def sim() -> SimCore:
    os.environ.setdefault("MUJOCO_GL", "egl")
    cfg = load_config(SCENE_YAML)
    return SimCore(cfg, str(PANDA), generate_scene(cfg, 1))


def test_starts_parked_with_ros_joint_names(sim: SimCore) -> None:
    names, pos, _ = sim.joint_state()
    assert names[:7] == [f"panda_joint{i}" for i in range(1, 8)]
    assert names[7:] == ["panda_finger_joint1", "panda_finger_joint2"]
    assert np.allclose(pos[:7], PARK_POSE, atol=0.02)


def test_items_rest_on_table(sim: SimCore) -> None:
    half = sim.scene_cfg.item_height / 2
    for pos, _ in sim.item_poses().values():
        assert pos[2] == pytest.approx(half, abs=0.002)


def test_trajectory_is_followed(sim: SimCore) -> None:
    names = sim.arm_joint_names()
    start = np.array(PARK_POSE)
    goal = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.785])
    times = np.linspace(0.2, 2.0, 10)
    pts = [start + (goal - start) * (t / 2.0) for t in times]
    sim.start_trajectory(names, times, pts)
    while not sim.trajectory_done():
        sim.step(10)
    sim.step(int(0.5 / sim.timestep))  # settle
    assert sim.arm_error(names, goal) < 0.01  # rad


def test_bad_trajectory_rejected(sim: SimCore) -> None:
    with pytest.raises(ValueError, match="Unknown joints"):
        sim.start_trajectory(["elbow"], [1.0], [[0.0]])
    with pytest.raises(ValueError, match="increasing"):
        sim.start_trajectory(["panda_joint1"], [1.0, 0.5], [[0.0], [0.1]])


def test_gripper_closes_fully_on_nothing(sim: SimCore) -> None:
    sim.set_gripper(0.0)
    sim.step(int(1.0 / sim.timestep))
    pos, vel = sim.finger_state()
    assert pos < 0.002 and abs(vel) < 1e-3
    sim.set_gripper(FINGER_MAX)
    sim.step(int(1.0 / sim.timestep))
    assert sim.finger_state()[0] > 0.035


def test_reset_scene_rebuilds_items(sim: SimCore) -> None:
    v = sim.version
    sim.reset_scene(Scene(9, [Item("only", "blue", "large", "cube", 0.45, 0.0)]))
    assert sim.version == v + 1
    assert list(sim.item_poses()) == ["only"]


def test_sim_time_continues_across_resets(sim: SimCore) -> None:
    """/clock must never go backwards when a new scene is loaded."""
    sim.step(50)
    t_before = sim.time
    sim.reset_scene(generate_scene(sim.scene_cfg, 7))
    assert sim.time >= t_before  # never backwards (settling happens off the clock)


def test_failed_reset_changes_nothing(sim: SimCore) -> None:
    """A scene that can't be built must leave the running one intact (review M3-1)."""
    before = (sim.version, [it.id for it in sim.scene.items], sim.model.nbody)
    good_xml = sim.panda_xml
    sim.panda_xml = "/nonexistent/panda.xml"
    with pytest.raises(FileNotFoundError):
        sim.reset_scene(Scene(9, [Item("only", "blue", "large", "cube", 0.45, 0.0)]))
    sim.panda_xml = good_xml
    assert (sim.version, [it.id for it in sim.scene.items], sim.model.nbody) == before
    assert [it.id for it, _, _ in sim.items_with_poses()] == before[1]
    sim.step(10)  # still runs


def test_snapshot_matches_its_model(sim: SimCore) -> None:
    v1, m1, d1 = sim.snapshot()
    sim.reset_scene(Scene(9, [Item("only", "blue", "large", "cube", 0.45, 0.0)]))
    v2, m2, d2 = sim.snapshot()
    assert v2 == v1 + 1 and m2 is not m1
    assert d1.qpos.size == m1.nq and d2.qpos.size == m2.nq


def test_goal_tokens(sim: SimCore) -> None:
    names = sim.arm_joint_names()
    v1, t1 = sim.start_trajectory(names, [1.0], [list(PARK_POSE)])
    v2, t2 = sim.start_trajectory(names, [1.0], [list(PARK_POSE)])
    assert v1 == v2 == sim.version and t2 == t1 + 1 == sim.trajectory_id
    sim.cancel_trajectory(t1)  # stale id: must not stop the newer trajectory
    assert not sim.trajectory_done()
    _, g1 = sim.set_gripper(0.0)
    sim.hold_gripper(g1 - 1)  # stale: ignored
    assert sim.gripper_id == g1


@pytest.mark.parametrize(
    ("names", "times", "points"),
    [
        ([], [0.0, 1.0], [[], []]),  # review M3-r2: froze physics (IndexError every step)
        (["panda_joint1"], [], []),
        (["panda_joint1"], [1.0], [[float("nan")]]),  # review M3-r2: reported SUCCEEDED
        (["panda_joint1"], [1.0], [[float("inf")]]),
        (["panda_joint1"], [float("inf")], [[0.0]]),
        (["panda_joint1", "panda_joint1"], [1.0], [[0.0, 0.0]]),
        (["panda_joint1"], [-1.0], [[0.0]]),
    ],
)
def test_malformed_trajectories_rejected(
    sim: SimCore, names: list[str], times: list[float], points: list[list[float]]
) -> None:
    before = sim.trajectory_id
    with pytest.raises(ValueError):
        sim.start_trajectory(names, times, points)
    assert sim.trajectory_id == before
    sim.step(10)  # physics still runs


def test_nan_velocities_rejected(sim: SimCore) -> None:
    with pytest.raises(ValueError):
        sim.start_trajectory(["panda_joint1"], [1.0], [[0.0]], [[float("nan")]])


def test_nan_gripper_rejected(sim: SimCore) -> None:
    with pytest.raises(ValueError):
        sim.set_gripper(float("nan"))
