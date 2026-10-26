"""Node-level test for rule_learner: demos in -> state, question, task file out."""

import os
import time
from pathlib import Path

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sorting_interfaces.msg import Demonstration, Item, RuleState
from sorting_interfaces.srv import RemoveDemo
from sorting_task import task_file
from std_srvs.srv import Trigger

from rule_learner.node import RuleLearnerNode

LATCHED = QoSProfile(
    depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
QUERY = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


@pytest.fixture(autouse=True)
def _isolated_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROS_DOMAIN_ID", str(150 + os.getpid() % 50))


def _demo(i: int, color: str, size: str, shape: str, box: str, session: str = "s1") -> Demonstration:
    d = Demonstration(session_id=session, index=i, box=box, source="ar")
    d.item = Item(id=f"item_{i}", color=color, size=size, shape=shape)
    return d


def test_learns_asks_reports_conflicts_and_saves(tmp_path: Path) -> None:
    rclpy.init(args=["--ros-args", "-p", f"tasks_dir:={tmp_path}"])
    states: list[RuleState] = []
    queries: list[Item] = []
    try:
        node = RuleLearnerNode()
        probe = rclpy.create_node("probe", use_global_arguments=False)
        pub = probe.create_publisher(Demonstration, "/teach/demonstrations", LATCHED)
        probe.create_subscription(RuleState, "/rule_learner/state", states.append, LATCHED)
        probe.create_subscription(Item, "/teach/query", queries.append, QUERY)
        client = probe.create_client(Trigger, "/rule_learner/confirm")
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(probe)

        def spin(seconds: float) -> None:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                ex.spin_once(timeout_sec=0.02)

        demos = [
            ("red", "small", "cube", "A"),
            ("blue", "large", "prism", "B"),
            ("green", "medium", "cylinder", "C"),
            ("yellow", "small", "prism", "C"),
            ("red", "large", "cylinder", "A"),
            ("blue", "small", "cube", "B"),
        ]
        for i, (c, s, sh, b) in enumerate(demos, start=1):
            pub.publish(_demo(i, c, s, sh, b))
        spin(1.5)
        last = states[-1]
        assert last.n_demonstrations == 6 and last.session_id == "s1"
        assert last.candidates[0].description == "Sort by color: red to A, blue to B, others to C."
        assert last.candidates[0].family == "mapping"
        assert queries and queries[-1].color in ("red", "green", "blue", "yellow")

        pub.publish(_demo(7, "red", "small", "cube", "C"))  # contradicts demo 1
        spin(1.0)
        assert [(c.demo_a, c.demo_b) for c in states[-1].conflicts] == [(1, 7)]

        future = client.call_async(Trigger.Request())
        spin(1.0)
        assert future.done() and future.result().success
        saved = list(tmp_path.glob("*.json"))
        assert len(saved) == 1
        tf = task_file.load(saved[0])
        assert tf.rule.default_box == "C" and tf.provenance.demonstrations == 7

        pub.publish(_demo(1, "green", "small", "cube", "A", session="s2"))  # new session resets
        spin(1.0)
        assert states[-1].session_id == "s2" and states[-1].n_demonstrations == 1
        node.destroy_node()
        probe.destroy_node()
    finally:
        rclpy.shutdown()


def _setup(tmp_path: Path, extra: list[str] | None = None):  # noqa: ANN202
    args = ["--ros-args", "-p", f"tasks_dir:={tmp_path}", *[x for e in (extra or []) for x in ("-p", e)]]
    rclpy.init(args=args)
    node = RuleLearnerNode()
    probe = rclpy.create_node("probe", use_global_arguments=False)
    seen: dict = {"states": [], "queries": []}
    pub = probe.create_publisher(Demonstration, "/teach/demonstrations", LATCHED)
    probe.create_subscription(RuleState, "/rule_learner/state", seen["states"].append, LATCHED)
    probe.create_subscription(Item, "/teach/query", seen["queries"].append, QUERY)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)

    def spin(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.02)

    return node, probe, pub, seen, spin


def _call(probe, spin, srv_type, name, request):  # noqa: ANN001, ANN202
    client = probe.create_client(srv_type, name)
    fut = client.call_async(request)
    spin(1.0)
    assert fut.done()
    return fut.result()


OPENING = [
    ("red", "small", "cube", "A"),
    ("blue", "large", "prism", "B"),
    ("green", "medium", "cylinder", "C"),
    ("yellow", "small", "prism", "C"),
    ("red", "large", "cylinder", "A"),
    ("blue", "small", "cube", "B"),
]


def test_reset_then_new_question_has_a_fresh_id(tmp_path: Path) -> None:
    node, probe, pub, seen, spin = _setup(tmp_path)
    try:
        for i, d in enumerate(OPENING, start=1):
            pub.publish(_demo(i, *d))
        spin(1.2)
        first = [q.id for q in seen["queries"]]
        assert first
        assert _call(probe, spin, Trigger, "/rule_learner/reset", Trigger.Request()).success
        assert seen["states"][-1].n_demonstrations == 0 and not seen["states"][-1].asking
        for i, d in enumerate(OPENING, start=11):
            pub.publish(_demo(i, *d))
        spin(1.2)
        ids = [q.id for q in seen["queries"]]
        assert len(ids) > len(first) and len(set(ids)) == len(ids)  # never reused
    finally:
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_no_question_before_opening_scene_and_duplicates_ignored(tmp_path: Path) -> None:
    node, probe, pub, seen, spin = _setup(tmp_path)
    try:
        for i, d in enumerate(OPENING[:3], start=1):
            pub.publish(_demo(i, *d))
        pub.publish(_demo(2, *OPENING[1]))  # latched re-delivery of #2
        spin(1.0)
        assert seen["queries"] == []  # ask_after_demos = 6
        assert seen["states"][-1].n_demonstrations == 3
    finally:
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_remove_demo_and_budget_exhaustion(tmp_path: Path) -> None:
    node, probe, pub, seen, spin = _setup(tmp_path, ["max_queries:=1", "query_timeout_s:=0.5"])
    try:
        for i, d in enumerate(OPENING, start=1):
            pub.publish(_demo(i, *d))
        spin(1.0)
        res = _call(probe, spin, RemoveDemo, "/rule_learner/remove_demo", RemoveDemo.Request(index=3))
        assert res.success and seen["states"][-1].n_demonstrations == 5
        assert not _call(
            probe, spin, RemoveDemo, "/rule_learner/remove_demo", RemoveDemo.Request(index=99)
        ).success
        spin(2.0)  # the single question goes unanswered; timeout re-asks, but the budget is 1
        st = seen["states"][-1]
        assert st.queries_asked <= 1
        node.learner.add(
            __import__("rule_learner.learner", fromlist=["Demo"]).Demo(
                50, {"color": "green", "size": "large", "shape": "prism"}, "A"
            )
        )
        node.n_queries = 1
        node.pending_query = None  # pretend the question was answered
        node._maybe_ask()
        node._maybe_ask()
        assert node.budget_exhausted
    finally:
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_no_more_questions_after_convergence(tmp_path: Path) -> None:
    node, probe, pub, seen, spin = _setup(tmp_path)
    try:
        colors = {"red": "A", "blue": "B", "green": "C", "yellow": "C"}
        i = 0
        for c, b in colors.items():
            for s in ("small", "medium", "large"):
                for sh in ("cube", "cylinder", "prism"):
                    i += 1
                    pub.publish(_demo(i, c, s, sh, b))
        spin(2.0)
        st = seen["states"][-1]
        assert st.converged and not st.asking
        n = len(seen["queries"])
        pub.publish(_demo(i + 1, "red", "small", "cube", "A"))
        spin(0.8)
        assert len(seen["queries"]) == n
    finally:
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_confirm_with_unwritable_dir_fails_gracefully(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    node, probe, pub, seen, spin = _setup(tmp_path, [f"tasks_dir:={blocker}/sub"])
    try:
        for i, d in enumerate(OPENING, start=1):
            pub.publish(_demo(i, *d))
        spin(1.0)
        res = _call(probe, spin, Trigger, "/rule_learner/confirm", Trigger.Request())
        assert not res.success and "Could not save the task file" in res.message
        assert node.context.ok()  # still alive
    finally:
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_timeout_resends_same_question_and_only_its_answer_clears_it(tmp_path: Path) -> None:
    node, probe, pub, seen, spin = _setup(tmp_path, ["query_timeout_s:=0.6"])
    try:
        for i, d in enumerate(OPENING, start=1):
            pub.publish(_demo(i, *d))
        spin(2.5)  # more than two timeouts
        ids = [q.id for q in seen["queries"]]
        assert len(ids) >= 2 and len(set(ids)) == 1  # re-sent with the same id, no new question
        assert node.n_queries == 1
        stale = _demo(20, "green", "large", "prism", "C")
        stale.source, stale.item.id = "query", "query_someoldid"
        pub.publish(stale)  # an answer to a different (old) question
        spin(0.5)
        assert node.pending_query is not None and node.pending_query.id == ids[0]
        q = seen["queries"][0]
        answer = _demo(21, q.color, q.size, q.shape, "A")
        answer.source, answer.item.id = "query", q.id
        pub.publish(answer)
        spin(0.5)
        assert node.pending_query is None or node.pending_query.id != ids[0]
    finally:
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()


def test_second_confirm_does_not_overwrite(tmp_path: Path) -> None:
    node, probe, pub, seen, spin = _setup(tmp_path)
    try:
        for i, d in enumerate(OPENING, start=1):
            pub.publish(_demo(i, *d))
        spin(1.0)
        a = _call(probe, spin, Trigger, "/rule_learner/confirm", Trigger.Request())
        b = _call(probe, spin, Trigger, "/rule_learner/confirm", Trigger.Request())
        assert a.success and b.success and a.message != b.message
        assert len(list(tmp_path.glob("*.json"))) == 2 and list(tmp_path.glob("*_2.json"))
    finally:
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()
