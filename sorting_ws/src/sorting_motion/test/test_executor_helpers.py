"""Executor helpers that need no robot: pose parsing, tray membership, summaries."""

from __future__ import annotations

import math

from geometry_msgs.msg import Pose
from sorting_interfaces.msg import Item as ItemMsg
from sorting_interfaces.msg import ItemPoses
from sorting_motion.executor import DEFAULT_RULE, hand_empty, items_from_poses, summarize, tray_of

from sorting_scene import default_config_path, load_config

CFG = load_config(default_config_path())


def _pose(x: float, y: float, z: float, yaw: float = 0.0) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    p.orientation.w, p.orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)
    return p


def test_items_from_poses_keeps_resting_items_with_yaw() -> None:
    msg = ItemPoses()
    msg.items = [
        ItemMsg(id="a", color="red", size="small", shape="cube"),
        ItemMsg(id="b", color="blue", size="large", shape="prism"),
    ]
    msg.poses = [_pose(0.4, 0.1, 0.0175, 0.5), _pose(0.5, 0.0, 0.2)]  # b is in the air (held)
    items = items_from_poses(msg, CFG)
    assert [i.id for i in items] == ["a"]
    assert items[0].yaw == pytest_approx(0.5)


def pytest_approx(v: float) -> object:
    import pytest

    return pytest.approx(v, abs=1e-9)


def test_tray_of() -> None:
    b = CFG.boxes[0]
    assert tray_of(CFG, b.x, b.y, 0.02) == b.label
    assert tray_of(CFG, b.x, b.y, 0.0565) == b.label  # stacked on another item, inside the walls
    assert tray_of(CFG, b.x, b.y, b.wall_height + CFG.item_height + 0.01) == ""  # above the tray
    assert tray_of(CFG, 0.45, -0.1, 0.0175) == ""  # on the table


def test_default_rule_sorts_by_color() -> None:
    assert DEFAULT_RULE.apply({"color": "red", "size": "small", "shape": "cube"}) == "A"
    assert DEFAULT_RULE.apply({"color": "blue", "size": "small", "shape": "cube"}) == "B"
    assert DEFAULT_RULE.apply({"color": "green", "size": "small", "shape": "cube"}) == "C"


def test_summarize_separates_infeasible_planning_failures_and_lost() -> None:
    def item(status: str, lifted: bool, final: str, target: str, reason: str = "") -> dict[str, object]:
        return {
            "status": status,
            "lifted": lifted,
            "final_box": final,
            "target": target,
            "reason": reason,
            "plan_ms": 5.0,
            "motion_s": 9.0,
            "planners": ["PTP", "LIN"] if status not in ("skipped", "lost") else [],
        }

    eps = [
        {
            "sim_s": 60.0,
            "items": [
                item("picked_and_released", True, "A", "A"),
                item("skipped", False, "", "B", "no_ik"),
                item("lost", False, "", "C", "not on the table at its turn"),
            ],
        },
        {
            "sim_s": 60.0,
            "items": [
                item("picked_and_released", True, "C", "B"),  # placed, wrong tray
                item("grasp_missed", False, "", "A"),
                item("skipped", False, "", "A", "plan_failed"),
            ],
        },
    ]
    s = summarize(eps)
    assert s["n_items"] == 6 and s["attempted"] == 3
    assert s["pick_success"] == round(2 / 3, 4)
    assert s["place_success"] == 0.5 and s["place_any_tray"] == 1.0
    assert s["sorting_accuracy"] == round(1 / 6, 4)
    assert s["sorting_accuracy_feasible"] == 0.2  # 1 correct of 5 (the no_ik item is excluded)
    assert s["infeasible"] == {"no_ik": 1}
    assert s["planning_failures"] == 1 and s["lost"] == 1
    assert s["failures"] == {"grasp_missed": 1}
    assert s["planners"] == {"PTP": 3, "LIN": 3}
    assert s["items_per_min_sim"] == 0.5


def test_items_below_the_table_are_not_on_it() -> None:
    msg = ItemPoses()
    msg.items = [ItemMsg(id="gone", color="red", size="small", shape="cube")]
    msg.poses = [_pose(3.06, 23.56, -57048.0)]  # seen in the M3 batch after a contact blow-up
    assert items_from_poses(msg, CFG) == []


def test_hand_empty_tolerates_a_tilted_item() -> None:
    # Measured in the grasp sweep: empty hand ~1.85 mm per finger; tilted small prism
    # (width 27.6 mm) held at ~9.9 mm per finger.
    assert hand_empty(0.00185, 0.0276)
    assert not hand_empty(0.0099, 0.0276)
    assert not hand_empty(0.0255, 0.052)  # large cylinder, normal grasp
