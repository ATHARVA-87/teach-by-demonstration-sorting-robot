"""Chain planning without a robot (D8 #6, #7): full plan, and every skip reason.

MoveIt runs offline in a subprocess (moveit_py crashes at interpreter exit,
troubleshooting #48) and reports JSON.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("moveit.planning")

SCRIPT = r"""
import json, os, sys
import numpy as np, rclpy
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from sorting_motion.moveit_setup import PARK_POSE, moveit_config
from sorting_motion.pick_place import PickPlaceError, PickPlacePlanner
from sorting_scene import Box, Item, default_config_path, load_config

cfg = load_config(default_config_path())
rclpy.init()
mp = MoveItPy(node_name="plan_test", config_dict=moveit_config(use_sim_time=False, offline=True))
planner = PickPlacePlanner(mp, cfg)
start = RobotState(mp.get_robot_model()); start.set_to_default_values()
start.set_joint_group_positions("panda_arm", np.array(PARK_POSE)); start.update()
box_a = cfg.box("A")
out = {}

def run(name, it, items, box):
    planner.set_world(items)
    try:
        plan = planner.plan(it, items, box, 0, start)
        out[name] = {"ok": True, "segments": [s.name for s in plan.segments],
                     "planners": [s.planner for s in plan.segments if s.planner]}
    except PickPlaceError as e:
        out[name] = {"ok": False, "reason": e.reason}

lone = Item("lone", "red", "medium", "cube", 0.45, -0.10, 0.3)
run("reachable", lone, [lone], box_a)
far = Item("far", "red", "small", "cube", 0.85, 0.0, 0.0)
run("out_of_reach", far, [far], box_a)
mid = Item("mid", "red", "large", "cube", 0.45, -0.10, 0.0)
walls = [Item(f"n{k}", "blue", "large", "cube", 0.45 + dx, -0.10 + dy, 0.0)
         for k, (dx, dy) in enumerate(((0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)))]
run("boxed_in", mid, [mid, *walls], box_a)
far_box = Box("Z", 0.95, 0.0, box_a.size, box_a.wall_height, 0)
run("box_out_of_reach", lone, [lone], far_box)
at_base = Item("base", "red", "small", "cube", 0.05, 0.0, 0.0)  # inside the arm's own footprint
run("at_the_base", at_base, [at_base], box_a)

# The carried item: lowering the hand until only the *item* (hanging 6.5 mm below
# the fingertips) would enter the table must fail the carried-item check.
import math
from geometry_msgs.msg import PoseStamped
from moveit.planning import PlanRequestParameters
from sorting_motion.grasps import TCP_OFFSET, hand_down_quaternion, neutral_hand_yaw
planner.set_world([])
arm = mp.get_planning_component("panda_arm")
lin = PlanRequestParameters(mp, "lin_params")
def lower_to(tcp_z):
    st = RobotState(mp.get_robot_model()); st.set_to_default_values()
    q = hand_down_quaternion(neutral_hand_yaw(0.45, -0.1))
    above = PoseStamped(); above.header.frame_id = "panda_link0"
    above.pose.position.x, above.pose.position.y, above.pose.position.z = 0.45, -0.1, 0.10 + TCP_OFFSET
    above.pose.orientation.w, above.pose.orientation.x, above.pose.orientation.y, above.pose.orientation.z = q
    assert st.set_from_ik("panda_arm", above.pose, "panda_hand", 1.0); st.update()
    goal = PoseStamped(); goal.header = above.header; goal.pose.orientation = above.pose.orientation
    goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = 0.45, -0.1, tcp_z + TCP_OFFSET
    arm.set_start_state(robot_state=st); arm.set_goal_state(pose_stamped_msg=goal, pose_link="panda_hand")
    r = arm.plan(single_plan_parameters=lin)
    return r.trajectory if r else None
carried = Item("held", "red", "medium", "cube", 0.0, 0.0, 0.0)
deep, shallow = lower_to(0.012), lower_to(0.04)  # item bottom 5.5 mm below / 22.5 mm above the table
out["robot_alone_plans_deep"] = deep is not None
planner.attach(carried)
out["deep_clear_with_item"] = planner.carried_path_clear(deep) if deep is not None else None
out["shallow_clear_with_item"] = planner.carried_path_clear(shallow) if shallow is not None else None
planner.detach()
out["deep_clear_after_detach"] = planner.carried_path_clear(deep) if deep is not None else None

# Neutral wrist: MoveIt's 'ready' with joint1 = 0.7 rad closes along neutral_hand_yaw - 135 deg.
st = RobotState(mp.get_robot_model()); st.set_to_default_values()
q = [0.7, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
st.set_joint_group_positions("panda_arm", np.array(q)); st.update()
o = st.get_pose("panda_hand").orientation
w, x, y, z = o.w, o.x, o.y, o.z
closing = math.atan2(1 - 2 * (x * x + z * z), 2 * (x * y - w * z))  # hand y axis = column 2 of R
expected = neutral_hand_yaw(math.cos(0.7), math.sin(0.7)) - 3 * math.pi / 4
out["neutral_wrist_error_deg"] = abs(math.degrees(math.remainder(closing - expected, math.pi)))
print("RESULT " + json.dumps(out), flush=True)
os._exit(0)
"""


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    script = tmp_path_factory.mktemp("plan") / "plan_check.py"
    script.write_text(SCRIPT)
    env = dict(os.environ, ROS_DOMAIN_ID=str(random.randint(30, 101)), FASTDDS_BUILTIN_TRANSPORTS="UDPv4")
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=180)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"no result; stderr tail:\n{out.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


def test_reachable_item_gets_the_whole_chain(results: dict[str, dict[str, object]]) -> None:
    r = results["reachable"]
    assert r["ok"]
    assert r["segments"] == [
        "open",
        "to_pregrasp",
        "approach",
        "close",
        "lift",
        "to_box",
        "lower",
        "release",
        "retreat",
    ]
    planners = r["planners"]
    assert isinstance(planners, list) and planners[1:3] == ["LIN", "LIN"] and planners[4:] == ["LIN", "LIN"]


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("out_of_reach", "outside_workspace"),
        ("boxed_in", "no_collision_free_grasp"),
        ("box_out_of_reach", "place_unreachable"),
        ("at_the_base", "no_ik"),
    ],
)
def test_infeasible_items_are_skipped_with_a_reason(
    results: dict[str, dict[str, object]], case: str, reason: str
) -> None:
    assert results[case] == {"ok": False, "reason": reason}


def test_carried_item_is_checked(results: dict[str, object]) -> None:
    """Review M3-2 blocker: the item in the hand used to be invisible to planning."""
    assert results["robot_alone_plans_deep"] is True  # the robot alone fits: only the item collides
    assert results["deep_clear_with_item"] is False
    assert results["shallow_clear_with_item"] is True
    assert results["deep_clear_after_detach"] is True


def test_neutral_wrist_convention_matches_moveit_fk(results: dict[str, object]) -> None:
    err = results["neutral_wrist_error_deg"]
    assert isinstance(err, float) and err < 0.5


def test_planner_config_file_matches_defaults() -> None:
    from sorting_motion.pick_place import PlannerConfig, load_planner_config

    cfg_file = Path(__file__).resolve().parents[1] / "config" / "pick_place.yaml"
    assert load_planner_config(cfg_file) == PlannerConfig()
