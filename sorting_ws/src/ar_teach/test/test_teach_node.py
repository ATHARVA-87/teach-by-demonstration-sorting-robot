"""Node-level tests for ar_teach.teach_node (no webcam or person needed).

Each test runs the real TeachNode in-process with parameter overrides and
an isolated ROS domain, spins it for a few seconds, and checks its outputs.
"""

import os
import time
from collections.abc import Callable

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from sorting_interfaces.msg import Demonstration, Item
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ar_teach.teach_node import TeachNode
from sorting_common.image_utils import IMAGE_QOS

LATCHED = QoSProfile(
    depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)


@pytest.fixture(autouse=True)
def _isolated_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROS_DOMAIN_ID", str(100 + os.getpid() % 100))


def _run(params: list[str], seconds: float, probe_setup: Callable[[Node, dict], None]) -> dict:
    """Start TeachNode with ``-p`` params plus a probe node; spin; return what the probe saw."""
    args = ["--ros-args", "-p", "fps:=15.0"] + [x for p in params for x in ("-p", p)]
    rclpy.init(args=args)
    seen: dict = {"status": [], "images": 0, "demos": []}
    try:
        node = TeachNode()
        probe = rclpy.create_node("probe", use_global_arguments=False)
        probe.create_subscription(
            String, "/ar_teach/status", lambda m: seen["status"].append(m.data), LATCHED
        )
        probe.create_subscription(
            Image, "/ar_teach/image", lambda m: seen.__setitem__("images", seen["images"] + 1), IMAGE_QOS
        )
        probe.create_subscription(Demonstration, "/teach/demonstrations", seen["demos"].append, LATCHED)
        probe_setup(probe, seen)
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(probe)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.02)
        seen["node"] = node
        node.destroy_node()
        probe.destroy_node()
    finally:
        rclpy.shutdown()
    return seen


def test_missing_model_and_webcam_do_not_crash() -> None:
    seen = _run(
        [
            "source:=webcam",
            "model_path:=/nonexistent/hand.task",
            "webcam.index:=99",
            "webcam.reopen_period_s:=0.5",
        ],
        3.0,
        lambda p, s: None,
    )
    assert seen["images"] > 10  # placeholder frames keep flowing
    node = seen["node"]
    assert "model" in node.faults and "webcam" in node.faults
    assert any("Hand model missing" in s for s in seen["status"])


def test_corrupt_model_file_does_not_crash(tmp_path) -> None:  # noqa: ANN001
    bad = tmp_path / "hand.task"
    bad.write_bytes(os.urandom(2048))
    seen = _run(
        ["source:=webcam", f"model_path:={bad}", "webcam.index:=99", "webcam.reopen_period_s:=0.5"],
        2.0,
        lambda p, s: None,
    )
    assert "could not be loaded" in seen["node"].faults["model"]
    assert seen["images"] > 5


def test_synthetic_demonstrations_follow_the_scripted_rule() -> None:
    seen = _run(["source:=synthetic", "fps:=60.0", "scene_seed:=3"], 6.0, lambda p, s: None)
    demos = seen["demos"]
    assert len(demos) >= 2
    rule = {"red": "A", "blue": "B"}
    assert all(d.box == rule.get(d.item.color, "C") for d in demos)
    assert len({d.session_id for d in demos}) == 1 and demos[0].session_id
    assert [d.index for d in demos] == list(range(1, len(demos) + 1))


def test_invalid_query_is_rejected_and_valid_one_is_placed() -> None:
    def setup(probe: Node, seen: dict) -> None:
        pub = probe.create_publisher(
            Item,
            "/teach/query",
            QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
            ),
        )
        seen["query_pub"] = pub
        probe.create_timer(
            0.5, lambda: pub.publish(Item(id="q_bad", color="purple", size="small", shape="cube"))
        )
        probe.create_timer(
            1.0, lambda: pub.publish(Item(id="q_ok", color="blue", size="large", shape="prism"))
        )

    seen = _run(["source:=synthetic", "fps:=15.0"], 2.5, setup)
    items = seen["node"].session.items
    assert "q_bad" not in items
    assert "q_ok" in items and items["q_ok"].is_query
    q = items["q_ok"].item
    assert (q.x, q.y) != (0.0, 0.0)  # placed in a free spot by the node


def test_new_scene_starts_a_new_session() -> None:
    def setup(probe: Node, seen: dict) -> None:
        client = probe.create_client(Trigger, "/ar_teach/new_scene")
        seen["first"] = None

        def call() -> None:
            if client.service_is_ready() and "future" not in seen and seen["demos"]:
                seen["sid_before"] = [d.session_id for d in seen["demos"]] or None
                seen["future"] = client.call_async(Trigger.Request())

        probe.create_timer(0.5, call)

    seen = _run(["source:=synthetic", "fps:=60.0", "scene_seed:=1"], 4.0, setup)
    assert seen["future"].done() and seen["future"].result().success
    node = seen["node"]
    assert node.scene_seed == 2
    ids = [d.session_id for d in seen["demos"]]
    assert seen["sid_before"], "expected demonstrations before new_scene (fps 60, 0.5 s)"
    assert ids[-1] == node.session_id  # new demos carry the new session id
    assert node.session_id not in seen["sid_before"]
    assert len(set(ids)) == 2
