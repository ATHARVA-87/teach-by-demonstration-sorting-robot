"""Scripted pick-and-place with ground-truth item poses (M3 baseline).

This is the **ground-truth baseline**: item positions come from the
simulator's debug topic. From M4 on, perception replaces that input and this
executor is the upper bound it is compared against. Nothing else here reads
ground truth at run time; the scoring at the end of an episode does, which is
what evaluation is for.

One episode:
1. reset the simulated scene (seed), the arm starts parked;
2. read the item poses, decide each item's box with the task file's rule;
3. for each item: plan the whole chain (``PickPlacePlanner``), then execute
   it. The grasp is checked from the gripper result (fingers stopped on the
   item, not closed on air). Items skipped because a neighbour blocked the
   fingers get a second chance at the end, when the table is emptier;
4. park and score from ground truth: pick success, place success, sorting
   accuracy, with skipped items and their reason codes listed separately.

Usage:
    ros2 run sorting_motion pick_place --seeds 0 1 2 [--task tasks/x.json] [--out results/pick_place]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from control_msgs.action import ParallelGripperCommand
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from sorting_interfaces.msg import ItemPoses
from sorting_interfaces.srv import ResetScene
from sorting_task.rules import Rule
from sorting_task.task_file import load as load_task

from sorting_motion.moveit_setup import PARK_POSE, moveit_config, wait_for_state_sync
from sorting_motion.pick_place import (
    PickPlaceError,
    PickPlacePlan,
    PickPlacePlanner,
    PlannerConfig,
    load_planner_config,
)
from sorting_scene import Item, SceneConfig, default_config_path, load_config

# Nothing between the fingers when they close to less than this share of the item
# width. Not "width minus a few mm": a held prism or cylinder can tilt in the grasp
# (finger gap 4-8 mm under its width) while still held, and treating that as a drop
# made the executor open the hand and cause one (M3 batch).
EMPTY_GAP_SHARE = 0.5


def hand_empty(finger: float, width: float) -> bool:
    """True if one finger's opening says nothing is held (an empty hand closes to ~2 mm)."""
    return 2 * finger < EMPTY_GAP_SHARE * width


@dataclass
class ItemResult:
    """What happened to one item (``status`` is set while executing, ``final_box`` by scoring)."""

    id: str
    color: str
    size: str
    shape: str
    target: str
    status: str = "pending"  # picked_and_released | grasp_missed | dropped | skipped | execution_failed
    reason: str = ""  # reason code when skipped / failed
    detail: str = ""  # human-readable details (e.g. which planning stage failed)
    planners: list[str] = field(default_factory=list)
    plan_ms: float = 0.0
    motion_s: float = 0.0
    lifted: bool = False  # scoring: item was off the table after the lift
    final_box: str = ""  # scoring: tray it ended in ("" = none)
    final_xyz: list[float] = field(default_factory=list)  # scoring: where it came to rest


class CellClient:
    """ROS side of the executor: scene reset, gripper action, debug poses."""

    def __init__(self) -> None:
        self.node = Node("pick_place_executor")
        self._latest: ItemPoses | None = None
        self._finger = 0.0
        self._sim_time = 0.0
        self.node.create_subscription(ItemPoses, "/sim/debug/item_poses", self._on_poses, 10)
        self.node.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.node.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.reset_cli = self.node.create_client(ResetScene, "/sorting_sim/reset_scene")
        self.gripper = ActionClient(self.node, ParallelGripperCommand, "/panda_hand_controller/gripper_cmd")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin.start()

    def _on_clock(self, msg: Clock) -> None:
        self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def sim_time(self) -> float:
        """Latest simulation time (s) from /clock."""
        return self._sim_time

    def _on_poses(self, msg: ItemPoses) -> None:
        self._latest = msg

    def _on_joints(self, msg: JointState) -> None:
        if "panda_finger_joint1" in msg.name:
            self._finger = msg.position[msg.name.index("panda_finger_joint1")]

    @staticmethod
    def _wait(fut: Future, timeout: float) -> Any:
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            raise TimeoutError("no answer from the cell")
        return fut.result()

    def reset_scene(self, seed: int, n_items: int = 0, timeout: float = 20.0) -> None:
        """Load scene ``seed`` in the simulator (arm parked); raises on failure."""
        if not self.reset_cli.wait_for_service(timeout_sec=timeout):
            raise TimeoutError("/sorting_sim/reset_scene not available; is sorting_sim running?")
        resp = self._wait(self.reset_cli.call_async(ResetScene.Request(seed=seed, n_items=n_items)), timeout)
        if not resp.success:
            raise RuntimeError(resp.message)
        self._latest = None

    def item_poses(self, timeout: float = 5.0) -> ItemPoses:
        """Latest ground-truth poses received after the last reset (debug topic)."""
        end = time.monotonic() + timeout
        while self._latest is None and time.monotonic() < end:
            time.sleep(0.02)
        if self._latest is None:
            raise TimeoutError("no /sim/debug/item_poses")
        return self._latest

    def fresh_item_poses(self, timeout: float = 5.0) -> ItemPoses:
        """Ground-truth poses from a message received after this call."""
        self._latest = None
        return self.item_poses(timeout)

    def finger(self) -> float:
        """Current opening of one finger (m), from /joint_states."""
        return self._finger

    def settled_item_poses(self, tol: float = 0.001, timeout: float = 5.0) -> ItemPoses:
        """Ground-truth poses once nothing moves: two readings ~0.1 s apart agree within ``tol`` (m).

        After a drop, items can still be bouncing; planning a grasp from such a
        pose put a finger on the item and launched it out of the scene.
        """
        end = time.monotonic() + timeout
        prev = self.fresh_item_poses()
        while True:
            time.sleep(0.1)
            cur = self.fresh_item_poses()
            moved = max(
                (
                    abs(a.position.x - b.position.x)
                    + abs(a.position.y - b.position.y)
                    + abs(a.position.z - b.position.z)
                    for a, b in zip(prev.poses, cur.poses, strict=False)
                ),
                default=0.0,
            )
            if (moved < tol and len(prev.poses) == len(cur.poses)) or time.monotonic() > end:
                return cur
            prev = cur

    def item_height(self, item_id: str) -> float | None:
        """Ground-truth z of one item, fresh (scoring only); None if unknown."""
        msg = self.fresh_item_poses()
        for it, pose in zip(msg.items, msg.poses, strict=True):
            if it.id == item_id:
                return float(pose.position.z)
        return None

    def grip(self, position: float, timeout: float = 10.0) -> ParallelGripperCommand.Result:
        """Run one gripper goal (per-finger opening) and return its result."""
        if not self.gripper.wait_for_server(timeout_sec=timeout):
            raise TimeoutError("gripper action not available")
        goal = ParallelGripperCommand.Goal()
        goal.command = JointState(name=["panda_finger_joint1"], position=[position])
        handle = self._wait(self.gripper.send_goal_async(goal), timeout)
        if not handle.accepted:
            raise RuntimeError("gripper goal rejected")
        return self._wait(handle.get_result_async(), timeout).result

    def close(self) -> None:
        """Stop spinning and destroy the node."""
        self.executor.shutdown()
        self.node.destroy_node()


def items_from_poses(msg: ItemPoses, cfg: SceneConfig) -> list[Item]:
    """Items resting on the table top with their ground-truth pose.

    Excludes items in the air (held), in a tray (above the table top) and below
    it (fell off the table, or out of the simulation after a contact blow-up).
    """
    out = []
    for it, pose in zip(msg.items, msg.poses, strict=True):
        z = pose.position.z
        if not (cfg.surface_z <= z <= cfg.surface_z + cfg.item_height):
            continue
        q = pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        out.append(Item(it.id, it.color, it.size, it.shape, pose.position.x, pose.position.y, yaw))
    return out


def tray_of(cfg: SceneConfig, x: float, y: float, z: float) -> str:
    """Tray label containing a resting point, or ''."""
    b = cfg.box_at(x, y)
    # Inside the walls, possibly stacked on other items (a full bin piles up).
    return b.label if b is not None and z < cfg.surface_z + b.wall_height + cfg.item_height else ""


class ScriptedExecutor:
    """Runs and scores pick-and-place episodes."""

    def __init__(
        self,
        moveit: MoveItPy,
        cell: CellClient,
        cfg: SceneConfig,
        rule: Rule,
        planner_cfg: PlannerConfig | None = None,
    ) -> None:
        self.moveit = moveit
        self.cell = cell
        self.cfg = cfg
        self.rule = rule
        self.planner = PickPlacePlanner(moveit, cfg, planner_cfg)

    def _current_state(self) -> RobotState:
        with self.moveit.get_planning_scene_monitor().read_only() as scene:
            st = RobotState(self.moveit.get_robot_model())
            st.set_to_default_values()
            st.set_joint_group_positions(
                "panda_arm", np.asarray(scene.current_state.get_joint_group_positions("panda_arm"))
            )
            st.update()
            return st

    def run_episode(self, seed: int, n_items: int = 0) -> dict[str, Any]:
        """Reset to scene ``seed``, sort every item, park, and return the scored episode."""
        self.cell.reset_scene(seed, n_items)
        if not wait_for_state_sync(self.moveit, self.cell.node):
            raise RuntimeError("MoveIt did not receive the robot state after the reset")
        items = items_from_poses(self.cell.item_poses(), self.cfg)
        results = {
            it.id: ItemResult(
                it.id,
                it.color,
                it.size,
                it.shape,
                self.rule.apply({"color": it.color, "size": it.size, "shape": it.shape}),
            )
            for it in items
        }
        slots: Counter[str] = Counter()
        t_start, sim_start = time.monotonic(), self.cell.sim_time()
        todo = [it.id for it in items]
        for attempt in (1, 2):  # second pass: items a neighbour blocked may be clear now
            retry = []
            for item_id in todo:
                r = results[item_id]
                box = self.cfg.box(r.target)
                # Look again before every pick (as a camera would): an earlier pick or
                # drop may have nudged items, and a stale pose means grasping air.
                on_table = {x.id: x for x in items_from_poses(self.cell.settled_item_poses(), self.cfg)}
                if item_id not in on_table:  # knocked into a tray, off the table or out of the sim
                    r.status, r.reason = "lost", "not on the table at its turn"
                    continue
                # Every item still on the table is an obstacle, including ones whose
                # pick failed earlier; what's still to do is tracked in ``todo``.
                obstacles = list(on_table.values())
                self.planner.set_world(obstacles)
                t0 = time.perf_counter()
                try:
                    plan = self.planner.plan(
                        on_table[item_id], obstacles, box, slots[box.label], self._current_state()
                    )
                except PickPlaceError as exc:
                    r.status, r.reason, r.detail = "skipped", exc.reason, exc.detail
                    if exc.reason == "no_collision_free_grasp" and attempt == 1:
                        retry.append(item_id)
                    continue
                r.plan_ms = (time.perf_counter() - t0) * 1e3
                r.planners = [s.planner for s in plan.segments if s.planner]
                r.reason = r.detail = ""
                try:
                    self._execute(plan, r)
                except Exception as exc:  # a timeout must not end the whole benchmark
                    r.status, r.reason = "execution_failed", f"{type(exc).__name__}: {exc}"
                    try:
                        self.cell.grip(plan.grasp.opening / 2)
                    except Exception:
                        pass
                if r.status == "picked_and_released":
                    slots[box.label] += 1
            todo = retry
        self._park()
        return self._score(seed, results, time.monotonic() - t_start, self.cell.sim_time() - sim_start)

    def _execute(self, plan: PickPlacePlan, r: ItemResult) -> None:
        t0 = self.cell.sim_time()  # motion time in sim time: independent of the real-time factor
        width = plan.grasp.width
        for seg in plan.segments:
            if seg.gripper is not None:
                res = self.cell.grip(seg.gripper)
                if seg.name == "close":
                    held = res.state.position[0] if res.state.position else self.cell.finger()
                    if hand_empty(held, width):  # closed past the item: nothing between the fingers
                        r.status, r.reason = (
                            "grasp_missed",
                            f"fingers at {2 * held * 1000:.1f} mm, item {width * 1000:.1f} mm",
                        )
                continue
            status = self.moveit.execute(seg.trajectory, controllers=[]).status
            if status != "SUCCEEDED":
                r.status, r.reason = "execution_failed", f"{seg.name}: {status}"
                break
            if seg.name == "lift":
                # Scoring only (ground truth): was the item really lifted?
                try:
                    z = self.cell.item_height(r.id)
                except TimeoutError:
                    z = None  # unknown counts as not lifted; the run goes on
                r.lifted = z is not None and z > self.cfg.surface_z + 0.05
                if r.status == "grasp_missed":
                    break  # nothing to carry; the arm is up and clear
            if seg.name in ("lift", "to_box", "lower") and hand_empty(self.cell.finger(), width):
                # Fingers closed past the item's width: it slipped out (run-time check, no ground truth).
                r.status, r.reason = "dropped", f"item slipped during '{seg.name}'"
                break
        else:
            r.status = "picked_and_released"
        r.motion_s = round(self.cell.sim_time() - t0, 2)
        if r.status in ("grasp_missed", "dropped", "execution_failed"):
            self.cell.grip(plan.grasp.opening / 2)  # let go; the world is re-read before the next item

    def _park(self) -> None:
        st = RobotState(self.moveit.get_robot_model())
        st.set_to_default_values()
        st.set_joint_group_positions("panda_arm", np.asarray(PARK_POSE))
        st.update()
        arm = self.moveit.get_planning_component("panda_arm")
        arm.set_start_state_to_current_state()
        arm.set_goal_state(robot_state=st)
        plan = arm.plan(single_plan_parameters=self.planner.ptp) or arm.plan(
            single_plan_parameters=self.planner.ompl
        )
        if plan:
            self.moveit.execute(plan.trajectory, controllers=[])

    def _score(
        self, seed: int, results: dict[str, ItemResult], wall_s: float, sim_s: float
    ) -> dict[str, Any]:
        time.sleep(0.5)  # let dropped items come to rest
        final = self.cell.fresh_item_poses()
        for it, pose in zip(final.items, final.poses, strict=True):
            if it.id in results:
                p = pose.position
                results[it.id].final_box = tray_of(self.cfg, p.x, p.y, p.z)
                results[it.id].final_xyz = [round(p.x, 4), round(p.y, 4), round(p.z, 4)]
        items = [asdict(r) for r in results.values()]
        return {"seed": seed, **score(items, sim_s), "wall_s": round(wall_s, 1), "items": items}


# Used when no task file is given: the rule from the M2 synthetic demo.
DEFAULT_RULE = Rule.from_parts([({"color": "red"}, "A"), ({"color": "blue"}, "B")], "C", "mapping")


# Skip reasons that mean "this item can't be done here", as opposed to a planner
# that failed on a feasible item ("plan_failed") or an item that was lost.
INFEASIBLE = ("outside_workspace", "no_collision_free_grasp", "no_ik", "place_unreachable")


def score(items: list[dict[str, Any]], sim_s: float) -> dict[str, Any]:
    """Metrics over item records (one episode or many). Rates are pooled over items.

    - pick success: lifted / attempted (attempted = planned and executed);
    - place success: in the *target* tray / lifted (``place_any_tray``: in any tray);
    - sorting accuracy: in the target tray / all items; ``_feasible``: over items
      that weren't skipped as infeasible;
    - infeasible (by reason), planning failures and lost items are counted separately;
    - throughput: items sorted per minute of simulated time.
    """
    attempted = [r for r in items if r["status"] not in ("skipped", "lost")]
    picked = [r for r in attempted if r["lifted"]]
    correct = [r for r in items if r["final_box"] == r["target"]]
    infeasible = [r for r in items if r["status"] == "skipped" and r["reason"] in INFEASIBLE]
    feasible_n = len(items) - len(infeasible)

    def rate(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    def median(key: str) -> float | None:
        return round(float(np.median([r[key] for r in attempted])), 1) if attempted else None

    return {
        "n_items": len(items),
        "attempted": len(attempted),
        "pick_success": rate(len(picked), len(attempted)),
        "place_success": rate(sum(r["final_box"] == r["target"] for r in picked), len(picked)),
        "place_any_tray": rate(sum(bool(r["final_box"]) for r in picked), len(picked)),
        "sorting_accuracy": rate(len(correct), len(items)),
        "sorting_accuracy_feasible": rate(len(correct), feasible_n),
        "infeasible": dict(Counter(r["reason"] for r in infeasible)),
        "planning_failures": sum(r["status"] == "skipped" and r["reason"] == "plan_failed" for r in items),
        "lost": sum(r["status"] == "lost" for r in items),
        "failures": dict(Counter(r["status"] for r in attempted if r["status"] != "picked_and_released")),
        "planners": dict(Counter(p for r in items for p in r["planners"])),
        "median_plan_ms": median("plan_ms"),
        "median_motion_s": median("motion_s"),
        "sim_s": round(sim_s, 1),
        "items_per_min_sim": round(len(correct) / (sim_s / 60), 2) if sim_s > 0 else None,
    }


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over episodes (rates pooled over items, not averaged over episodes)."""
    items = [r for e in episodes for r in e["items"]]
    return {"episodes": len(episodes), **score(items, sum(e["sim_s"] for e in episodes))}


def markdown_table(summary: dict[str, Any]) -> str:
    """The README results table, generated from ``summary.json`` (no hand-copied numbers)."""

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.1f}%"

    infeasible = sum(summary["infeasible"].values())
    correct = round((summary["sorting_accuracy"] or 0) * summary["n_items"])
    fallbacks = {k: v for k, v in summary["planners"].items() if k not in ("PTP", "LIN")}
    planners = ", ".join(f"{k} {v}" for k, v in summary["planners"].items())
    rows = [
        ("Scenes / items", f"{summary['episodes']} / {summary['n_items']}"),
        ("Pick success (lifted / attempted)", pct(summary["pick_success"])),
        ("Place success (in the target tray / lifted)", pct(summary["place_success"])),
        (
            "**Sorting accuracy** (in the correct tray / all items)",
            f"**{pct(summary['sorting_accuracy'])}** ({correct}/{summary['n_items']})",
        ),
        ("Sorting accuracy over feasible items", pct(summary["sorting_accuracy_feasible"])),
        ("Infeasible (skipped with a reason)", f"{infeasible} {summary['infeasible'] or ''}".strip()),
        ("Planning failures on feasible items", str(summary["planning_failures"])),
        ("Lost (left the table before their turn)", str(summary["lost"])),
        ("Execution failures", ", ".join(f"{k} {v}" for k, v in summary["failures"].items()) or "0"),
        ("Planners used (segments)", planners + ("" if fallbacks else " (no fallbacks)")),
        ("Planning per item, median", f"{summary['median_plan_ms']} ms"),
        ("Motion per item, median (sim time)", f"{summary['median_motion_s']} s"),
        ("Throughput (sim time)", f"{summary['items_per_min_sim']} items/min"),
    ]
    return "\n".join(["| Metric | Value |", "|---|---|", *(f"| {a} | {b} |" for a, b in rows)])


def main(argv: list[str] | None = None) -> None:
    """Entry point (needs a running ``sorting_sim``). Exits via ``os._exit`` (moveit_py teardown crash)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-items", type=int, default=0, help="0: random within scene.yaml's range")
    ap.add_argument("--task", type=Path, help="task file from the teach mode (default: sort by color)")
    ap.add_argument("--scene-config", type=Path, default=None)
    ap.add_argument("--planner-config", type=Path, default=None, help="default: installed pick_place.yaml")
    ap.add_argument("--out", type=Path, default=Path("results/pick_place"))
    ap.add_argument("--table-from", type=Path, help="print the results table from a summary.json and exit")
    args = ap.parse_args(argv)
    if args.table_from:
        print(markdown_table(json.loads(args.table_from.read_text())))
        return

    cfg = load_config(args.scene_config or default_config_path())
    rule = load_task(args.task).to_rule() if args.task else DEFAULT_RULE
    rclpy.init()
    moveit = MoveItPy(node_name="pick_place_moveit", config_dict=moveit_config())
    cell = CellClient()
    code = 0
    try:
        runner = ScriptedExecutor(moveit, cell, cfg, rule, load_planner_config(args.planner_config))
        args.out.mkdir(parents=True, exist_ok=True)
        episodes = []
        for seed in args.seeds:
            ep = runner.run_episode(seed, args.n_items)
            episodes.append(ep)
            (args.out / f"episode_{seed:04d}.json").write_text(json.dumps(ep, indent=1))
            print(
                f"seed {seed}: {ep['n_items']} items, pick {ep['pick_success']}, "
                f"place {ep['place_success']}, sorted {ep['sorting_accuracy']}, "
                f"infeasible {ep['infeasible']}, lost {ep['lost']}, {ep['wall_s']} s",
                flush=True,
            )
        summary = summarize(episodes)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        print(f"pick_place failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        code = 1
    cell.close()
    rclpy.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # moveit_py crashes in its static destructors at exit (troubleshooting #48)


if __name__ == "__main__":
    main()
