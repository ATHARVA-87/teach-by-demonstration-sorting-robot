"""rule_learner node: learns the sorting rule from demonstrations, live.

Topics in:
    /teach/demonstrations     sorting_interfaces/Demonstration (reliable, latched)
Topics out:
    /rule_learner/state       sorting_interfaces/RuleState (latched): candidates with
                              probabilities, conflicts, query, converged flag
    /teach/query              sorting_interfaces/Item: the next question for ar_teach
Services:
    ~/confirm                 std_srvs/Trigger: save the current best rule as a task file
    ~/reset                   std_srvs/Trigger: forget all demonstrations
    ~/remove_demo             sorting_interfaces/RemoveDemo: delete one demonstration
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from pydantic import ValidationError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sorting_interfaces.msg import Conflict as ConflictMsg
from sorting_interfaces.msg import Demonstration, RuleCandidate, RuleState
from sorting_interfaces.msg import Item as ItemMsg
from sorting_interfaces.srv import RemoveDemo
from sorting_task import task_file
from std_srvs.srv import Trigger

from rule_learner.hypotheses import FAMILIES, HypothesisSpace
from rule_learner.learner import Demo, LearnerConfig, RuleLearner, item_phrase
from sorting_common.node_runner import run_node
from sorting_scene import default_config_path, load_config

LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
QUERY_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
DEMO_QOS = QoSProfile(
    depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)


class RuleLearnerNode(Node):
    """Wraps :class:`RuleLearner` in ROS topics and services."""

    def __init__(self) -> None:
        super().__init__("rule_learner")
        p = self.declare_parameter
        scene_yaml = str(p("scene_config", "").value) or str(default_config_path())
        self.cfg = LearnerConfig(
            prior_lambda=float(p("prior_lambda", 1.0).value),
            epsilon=float(p("epsilon", 0.05).value),
            stop_confidence=float(p("stop_confidence", 0.99).value),
            max_queries=int(p("max_queries", 20).value),
            conflict_requery=str(p("conflict_requery", "unresolved").value),
            max_conflict_asks=int(p("max_conflict_asks", 4).value),
        )
        self.query_timeout_s = float(p("query_timeout_s", 60.0).value)
        # 0 = the scene's minimum size (scene.yaml n_items.min): the worker
        # sorts at least that many opening items before the first question.
        self.ask_after_demos = int(p("ask_after_demos", 0).value)
        families = list(p("families", list(FAMILIES)).value)
        self.active = bool(p("active_learning", True).value)
        self.top_k = int(p("candidates", 5).value)
        self.tasks_dir = Path(str(p("tasks_dir", "tasks").value)).expanduser()
        self.query_seed = int(p("query_seed", 0).value)

        self.scene_cfg = load_config(scene_yaml)
        space = HypothesisSpace.build(self.scene_cfg.attribute_values, self.scene_cfg.box_labels, families)
        self.learner = RuleLearner(space, self.cfg, seed=self.query_seed)
        if self.ask_after_demos <= 0:
            self.ask_after_demos = self.scene_cfg.n_items_min
        self.pending_query: ItemMsg | None = None
        self.n_queries = 0
        self.seen: set[tuple[str, int]] = set()  # (session_id, index) already applied
        self.session_id = ""
        self.budget_exhausted = False
        self._query_sent_at = 0.0

        self.state_pub = self.create_publisher(RuleState, "/rule_learner/state", LATCHED)
        # Volatile (not latched): a restarted ar_teach must not receive a stale
        # question; a lost question is re-asked after query_timeout_s instead.
        self.query_pub = self.create_publisher(ItemMsg, "/teach/query", QUERY_QOS)
        self.create_timer(1.0, self._check_pending_query)
        self.create_subscription(Demonstration, "/teach/demonstrations", self._on_demo, DEMO_QOS)
        self.create_service(Trigger, "~/confirm", self._on_confirm)
        self.create_service(Trigger, "~/reset", self._on_reset)
        self.create_service(RemoveDemo, "~/remove_demo", self._on_remove)
        self.get_logger().info(
            f"Rule learner ready: {len(space)} hypotheses, active learning {'on' if self.active else 'off'}"
        )
        self._publish_state()

    # --- callbacks ------------------------------------------------------------
    def _on_demo(self, msg: Demonstration) -> None:
        if msg.session_id != self.session_id:
            if self.learner.demos:
                self.get_logger().info(f"New teaching session {msg.session_id}: starting over.")
            self._clear()
            self.session_id = msg.session_id
        if (msg.session_id, msg.index) in self.seen:
            return  # latched topic re-delivery after a restart
        self.seen.add((msg.session_id, msg.index))
        attrs = {"color": msg.item.color, "size": msg.item.size, "shape": msg.item.shape}
        try:
            self.learner.add(Demo(int(msg.index), attrs, msg.box, msg.source))
        except ValueError as exc:
            self.get_logger().error(f"Ignored demonstration #{msg.index}: {exc}")
            return
        if self.pending_query is not None and msg.item.id == self.pending_query.id:
            self.pending_query = None  # only the answer to *this* question clears it
        best = self.learner.best()
        self.get_logger().info(
            f"Demonstration #{msg.index} ({msg.item.color} {msg.item.shape} {msg.item.size} "
            f"-> {msg.box}). Best rule: {best.description} ({best.probability:.0%})"
        )
        for c in self.learner.conflicts():
            self.get_logger().warning(c.message)
        self._maybe_ask()
        self._publish_state()

    def _maybe_ask(self) -> None:
        """Publish the next question unless confident, already asking, or out of budget."""
        if not self.active or self.pending_query is not None or self.learner.converged():
            return
        if len(self.learner.demos) < self.ask_after_demos:
            return  # let the worker sort the opening scene first (matches the benchmark protocol)
        if self.n_queries >= self.cfg.max_queries:
            if self.budget_exhausted:
                return
            self.budget_exhausted = True
            top = self.learner.candidates(2)
            if len(top) == 2:
                self.get_logger().warning(
                    f"Question budget used up. Still unsure between "
                    f"'{top[0].description}' ({top[0].probability:.0%}) and "
                    f"'{top[1].description}' ({top[1].probability:.0%}). "
                    "Review and correct the rule, or teach more."
                )
            return
        q = self.learner.best_query()
        self.n_queries += 1
        # Only the attributes are decided here; ar_teach places the item in a
        # free spot of its current scene (x = y = 0 means "you choose").
        msg = ItemMsg(
            id=f"query_{uuid.uuid4().hex[:10]}",  # unique across resets and restarts
            color=q.attributes["color"],
            size=q.attributes["size"],
            shape=q.attributes["shape"],
            x=0.0,
            y=0.0,
            yaw=0.0,
        )
        self.pending_query = msg
        self._query_sent_at = time.monotonic()
        self.query_pub.publish(msg)
        why = "to resolve a conflict" if q.reason == "conflict" else f"{q.information_bits:.2f} bits expected"
        self.get_logger().info(f"Asking about a {item_phrase(q.attributes)} ({why})")

    def _publish_state(self) -> None:
        st = RuleState()
        st.header.stamp = self.get_clock().now().to_msg()
        st.session_id = self.session_id
        st.n_demonstrations = len(self.learner.demos)
        st.queries_asked = self.n_queries
        st.budget_exhausted = self.budget_exhausted
        if self.learner.demos:
            for c in self.learner.candidates(self.top_k):
                st.candidates.append(
                    RuleCandidate(
                        description=c.description,
                        rule_json=json.dumps(c.rule.to_dict()),
                        family=c.rule.family,
                        probability=c.probability,
                        errors=c.errors,
                    )
                )
        st.conflicts = [
            ConflictMsg(demo_a=c.a, demo_b=c.b, message=c.message) for c in self.learner.conflicts()
        ]
        st.unexplained = [int(i) for i in self.learner.unexplained()] if self.learner.demos else []
        st.version_space_size = self.learner.version_space_size()
        st.converged = self.learner.converged()
        st.asking = self.pending_query is not None
        if self.pending_query is not None:
            st.query = self.pending_query
        self.state_pub.publish(st)

    def _on_confirm(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        if not self.learner.demos:
            resp.success, resp.message = False, "No demonstrations yet. Sort a few items first."
            return resp
        best = self.learner.best()
        name = re.sub(r"[^a-z0-9]+", "_", best.description.lower()).strip("_")[:60]
        attrs = {k: list(v) for k, v in self.scene_cfg.attribute_values.items()}
        try:
            prov = task_file.ProvenanceModel.model_validate(
                {
                    "approved": True,
                    "confidence": best.probability,
                    "demonstrations": len(self.learner.demos),
                    "queries": self.n_queries,
                    "epsilon": self.cfg.epsilon,
                    "lambda": self.cfg.prior_lambda,
                    "threshold": self.cfg.stop_confidence,
                    "competing": [
                        {"description": c.description, "probability": c.probability}
                        for c in self.learner.candidates(self.top_k)[1:]
                    ],
                    "unexplained_demonstrations": self.learner.unexplained(),
                }
            )
            tf = task_file.build_task_file(
                best.rule, name, best.description, attrs, list(self.scene_cfg.box_labels), prov
            )
        except ValidationError as exc:
            resp.success, resp.message = (
                False,
                f"The rule did not pass task-file validation: {exc.errors()[0]['msg']}",
            )
            self.get_logger().error(resp.message)
            return resp
        try:
            path = self.tasks_dir / f"{name}.json"
            n = 2
            while path.exists():  # never overwrite an approved task
                path = self.tasks_dir / f"{name}_{n}.json"
                n += 1
            path = task_file.save(tf, path)
        except OSError as exc:
            resp.success = False
            resp.message = (
                f"Could not save the task file in '{self.tasks_dir}' ({exc.strerror}). "
                "Check that the folder is writable or set the tasks_dir parameter."
            )
            self.get_logger().error(resp.message)
            return resp
        resp.success, resp.message = True, f"Saved task file {path}"
        self.get_logger().info(resp.message)
        return resp

    def _check_pending_query(self) -> None:
        """Clear a question that is no longer needed, or re-ask one that got lost."""
        if self.pending_query is None:
            return
        if self.learner.converged():
            self.pending_query = None  # the "?" item stays sortable; we just stop waiting
            self._publish_state()
        elif time.monotonic() - self._query_sent_at > self.query_timeout_s:
            # Re-send the *same* question (same id): ar_teach ignores it if the
            # item is still on the table, and places it if it was lost or
            # rejected. No new question, no budget change.
            self.get_logger().warning(
                f"No answer to question {self.pending_query.id} after {self.query_timeout_s:.0f} s. "
                "Sending it again in case the AR view missed it."
            )
            self._query_sent_at = time.monotonic()
            self.query_pub.publish(self.pending_query)

    def _clear(self) -> None:
        """Forget the current teaching session."""
        self.learner.reset()
        self.pending_query = None
        self.n_queries = 0
        self.budget_exhausted = False
        self.seen.clear()

    def _on_reset(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self._clear()
        self._publish_state()
        resp.success, resp.message = True, "All demonstrations cleared."
        return resp

    def _on_remove(self, req: RemoveDemo.Request, resp: RemoveDemo.Response) -> RemoveDemo.Response:
        if req.index not in {d.index for d in self.learner.demos}:
            resp.success, resp.message = False, f"Demonstration #{req.index} doesn't exist."
            return resp
        self.learner.remove(int(req.index))
        self._maybe_ask()
        self._publish_state()
        resp.success, resp.message = True, f"Removed demonstration #{req.index}."
        return resp


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run rule_learner rule_learner_node``."""
    run_node(RuleLearnerNode, args)


if __name__ == "__main__":
    main()
