"""Controller contract check (decision D8 #3): can MoveIt drive this robot?

Plans with MoveIt and executes through the robot's
``/panda_arm_controller/follow_joint_trajectory`` and
``/panda_hand_controller/gripper_cmd`` actions, then checks that the arm
settled on each goal. Works against any backend that serves the contract:
``sorting_sim``, mujoco_ros2_control + ros2_controllers, or a real Panda.

Usage (with a robot backend running):
    ros2 run sorting_motion contract_check

Prints one JSON line with the per-move results; exit code 0 = contract holds.
"""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import time
from typing import Any

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters

from sorting_motion.moveit_setup import PARK_POSE, moveit_config, wait_for_state_sync

SETTLED_TOL = 0.005  # rad


def run_check(moveit: MoveItPy, node: Any) -> dict[str, Any]:
    """Run the moves and return a report (``ok`` is the overall verdict)."""
    report: dict[str, Any] = {"ok": False, "moves": []}
    if not wait_for_state_sync(moveit, node):
        report["error"] = "MoveIt never received matching /joint_states"
        return report
    arm = moveit.get_planning_component("panda_arm")
    park = RobotState(moveit.get_robot_model())
    park.set_to_default_values()
    park.set_joint_group_positions("panda_arm", np.array(PARK_POSE))
    park.update()
    psm = moveit.get_planning_scene_monitor()
    ok = True
    for goal in ("ready", "park"):
        arm.set_start_state_to_current_state()
        if goal == "park":
            arm.set_goal_state(robot_state=park)
        else:
            arm.set_goal_state(configuration_name=goal)
        plan = arm.plan(single_plan_parameters=PlanRequestParameters(moveit, "ptp_params"))
        move: dict[str, Any] = {"group": "panda_arm", "goal": goal, "planned": bool(plan)}
        if plan:
            t0 = time.monotonic()
            move["status"] = moveit.execute(plan.trajectory, controllers=[]).status
            move["wall_s"] = round(time.monotonic() - t0, 2)
            move["duration_s"] = round(plan.trajectory.duration, 2)
            last = plan.trajectory[len(plan.trajectory) - 1].get_joint_group_positions("panda_arm")
            with psm.read_only() as scene:
                now = scene.current_state.get_joint_group_positions("panda_arm")
            move["final_error_mrad"] = round(
                float(np.max(np.abs(np.asarray(now) - np.asarray(last)))) * 1e3, 2
            )
        move["ok"] = (
            move.get("status") == "SUCCEEDED" and move.get("final_error_mrad", 1e9) < SETTLED_TOL * 1e3
        )
        ok &= move["ok"]
        report["moves"].append(move)

    hand = moveit.get_planning_component("hand")
    for goal in ("close", "open"):  # ParallelGripperCommand through MoveIt
        hand.set_start_state_to_current_state()
        hand.set_goal_state(configuration_name=goal)
        plan = hand.plan(single_plan_parameters=PlanRequestParameters(moveit, "ompl_params"))
        move = {"group": "hand", "goal": goal, "planned": bool(plan)}
        if plan:
            move["status"] = moveit.execute(plan.trajectory, controllers=[]).status
        move["ok"] = move.get("status") == "SUCCEEDED"
        ok &= move["ok"]
        report["moves"].append(move)
    report["ok"] = ok
    return report


def main() -> None:
    """Entry point. Exits via ``os._exit``: see the note below."""
    # Diagnostics: SORTING_HANG_DUMP_S=N prints every thread's stack if still running after N s.
    hang_dump = float(os.environ.get("SORTING_HANG_DUMP_S", "0"))
    if hang_dump > 0:
        faulthandler.dump_traceback_later(hang_dump, exit=False)
    rclpy.init()
    moveit = MoveItPy(node_name="contract_check_moveit", config_dict=moveit_config())
    node = rclpy.create_node("contract_check")
    try:
        report = run_check(moveit, node)
    except Exception as exc:  # report, don't hang
        report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    # moveit_py (2.15, Python 3.14) segfaults in its static destructors at
    # interpreter exit, even for a script that only constructs MoveItPy. All
    # work is done and flushed, so skip interpreter teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
