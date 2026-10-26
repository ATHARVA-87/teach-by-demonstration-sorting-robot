"""Plan a whole pick-and-place before the robot moves (decision D8 #6, #7).

For one item the chain is:

    start --PTP--> pre-grasp --LIN--> grasp  [close]  --LIN--> lift
          --PTP--> above slot --LIN--> release  [open]  --LIN--> above slot

Every arm segment is planned (each starting where the previous one ends)
before anything executes, so an unreachable box can't leave an item stuck in
the gripper. If a segment fails, the next grasp candidate is tried. Items that
can't be done are skipped with a reason code:

- ``outside_workspace``: beyond the arm's top-down reach;
- ``no_collision_free_grasp``: every finger placement would hit a neighbour;
- ``no_ik``: no joint solution for any clear grasp;
- ``plan_failed``: IK exists but MoveIt found no valid motion;
- ``place_unreachable``: the item could be picked but not placed.

Planners: Pilz PTP for transfers (OMPL RRTConnect as fallback), Pilz LIN for
the vertical approach/retreat (slower LIN, then a short PTP, as fallbacks).
Every segment records the planner that produced it.

Collisions: the planning scene holds the table, the tray walls and the other
items. MoveIt validates each segment for the *robot*. The *carried item* is
checked separately: it is attached to the hand in the scene, and every
waypoint of the carrying segments is checked against the scene state that
holds it (``carried_path_clear``). MoveIt's own validation never sees the
attachment here: the start states passed to the planner carry no attached
bodies, and MoveIt then drops the scene's attachments (M3 review).

Settings: ``config/pick_place.yaml`` (``PlannerConfig``).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from shape_msgs.msg import SolidPrimitive

from sorting_motion.grasps import (
    TCP_OFFSET,
    Grasp,
    candidates,
    finger_clearance,
    footprint_radius,
    hand_down_quaternion,
    neutral_hand_yaw,
    obstacles_for,
)
from sorting_motion.moveit_setup import READY_POSE, config_dir, within_limits
from sorting_scene import Box, Item, SceneConfig

FRAME = "panda_link0"
HAND = "panda_hand"
FINGERTIP_BELOW_TCP = 0.011  # m, lowest finger point below the TCP (Panda hand geometry)
TOUCH_LINKS = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
CARRIED_ID = "carried_item"


@dataclass(frozen=True)
class PlannerConfig:
    """Planner settings; defaults equal ``config/pick_place.yaml``."""

    reach_max: float = 0.80
    approach: float = 0.10
    release_clearance: float = 0.015
    clearance_min: float = 0.004
    joint_limit_margin: float = 0.03
    slot_offsets: tuple[tuple[float, float], ...] = (
        (-0.03, -0.03),
        (0.03, 0.03),
        (-0.03, 0.03),
        (0.03, -0.03),
    )


def load_planner_config(path: str | Path | None = None) -> PlannerConfig:
    """Read ``pick_place.yaml`` (default: the installed one); unknown keys are an error."""
    p = Path(path) if path else config_dir() / "pick_place.yaml"
    raw = yaml.safe_load(p.read_text()) or {}
    known = set(PlannerConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{p}: unknown keys {sorted(unknown)}")
    if "slot_offsets" in raw:
        raw["slot_offsets"] = tuple((float(a), float(b)) for a, b in raw["slot_offsets"])
    return PlannerConfig(**raw)


class PickPlaceError(Exception):
    """An item can't be done; ``reason`` is one of the module's reason codes."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class Segment:
    """One step of the chain: an arm trajectory or a gripper command."""

    name: str
    trajectory: Any = None  # moveit RobotTrajectory (arm segments)
    gripper: float | None = None  # finger opening per finger (gripper segments)
    planner: str = ""


@dataclass
class PickPlacePlan:
    """A fully planned chain for one item, ready to execute."""

    item_id: str
    box: str
    grasp: Grasp
    segments: list[Segment] = field(default_factory=list)
    end_state: Any = None  # RobotState at the end of the chain


def _pose(xyz: tuple[float, float, float], quat: tuple[float, float, float, float]) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = FRAME
    p = Pose()
    p.position.x, p.position.y, p.position.z = xyz
    p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = quat
    ps.pose = p
    return ps


def _box_object(
    name: str,
    size: tuple[float, float, float],
    xyz: tuple[float, float, float],
    yaw: float = 0.0,
    op: bytes = CollisionObject.ADD,
) -> CollisionObject:
    co = CollisionObject()
    co.header.frame_id = FRAME
    co.id = name
    co.operation = op
    prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(size))
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.w, pose.orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)
    co.primitives = [prim]
    co.primitive_poses = [pose]
    return co


def _item_object(it: Item, cfg: SceneConfig) -> CollisionObject:
    """Items as their bounding box (prisms/cylinders included: conservative and cheap)."""
    s = cfg.sizes[it.size]
    return _box_object(
        it.id, (s, s, cfg.item_height), (it.x, it.y, cfg.surface_z + cfg.item_height / 2), it.yaw
    )


class PickPlacePlanner:
    """Plans pick-and-place chains with moveit_py and maintains the planning scene."""

    def __init__(self, moveit: MoveItPy, cfg: SceneConfig, pc: PlannerConfig | None = None) -> None:
        self.moveit = moveit
        self.cfg = cfg
        self.pc = pc or PlannerConfig()
        self.arm = moveit.get_planning_component("panda_arm")
        self.model = moveit.get_robot_model()
        self.psm = moveit.get_planning_scene_monitor()
        self.ptp = PlanRequestParameters(moveit, "ptp_params")
        self.lin = PlanRequestParameters(moveit, "lin_params")
        self.ompl = PlanRequestParameters(moveit, "ompl_params")
        self.ptp_carry = PlanRequestParameters(moveit, "ptp_carry_params")
        self.lin_slow = PlanRequestParameters(moveit, "lin_slow_params")

    # --- planning scene -------------------------------------------------------
    def set_world(self, items: list[Item]) -> None:
        """Table, tray walls and ``items`` as collision objects (replaces the old world)."""
        cfg = self.cfg
        objs = [_box_object("table", (1.40, 1.24, 0.04), (0.30, 0.06, cfg.surface_z - 0.02))]
        for b in cfg.boxes:
            h, half, t = b.wall_height, b.size / 2, 0.004
            for k, (dx, dy, sx, sy) in enumerate(
                ((half, 0, t, b.size), (-half, 0, t, b.size), (0, half, b.size, t), (0, -half, b.size, t))
            ):
                objs.append(_box_object(f"box_{b.label}_wall{k}", (sx, sy, h), (b.x + dx, b.y + dy, h / 2)))
        objs += [_item_object(it, cfg) for it in items]
        with self.psm.read_write() as scene:
            scene.remove_all_collision_objects()
            for co in objs:
                scene.apply_collision_object(co)
            scene.current_state.update()

    def attach(self, it: Item) -> None:
        """Attach the carried item to the hand: a cylinder around its footprint, centred on the TCP.

        Defined in the ``panda_hand`` frame (grasped at mid-height, so the item's
        centre is at the TCP); a cylinder needs no yaw, and it covers every
        shape. A pose in the world frame would be converted with the scene's
        *current* arm state, not the grasp state.
        """
        co = CollisionObject()
        co.header.frame_id = HAND
        co.id = CARRIED_ID
        co.operation = CollisionObject.ADD
        radius = footprint_radius(it.shape, self.cfg.sizes[it.size])
        co.primitives = [
            SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[self.cfg.item_height, radius])
        ]
        pose = Pose()
        pose.position.z = TCP_OFFSET
        pose.orientation.w = 1.0
        co.primitive_poses = [pose]
        aco = AttachedCollisionObject(link_name=HAND, object=co, touch_links=TOUCH_LINKS)
        with self.psm.read_write() as scene:
            scene.process_attached_collision_object(aco)

    def detach(self) -> None:
        """Remove the carried item from the hand (and from the scene)."""
        aco = AttachedCollisionObject(link_name=HAND)
        aco.object.id = CARRIED_ID
        aco.object.operation = CollisionObject.REMOVE
        with self.psm.read_write() as scene:
            scene.process_attached_collision_object(aco)
            rm = CollisionObject(id=CARRIED_ID, operation=CollisionObject.REMOVE)
            rm.header.frame_id = FRAME
            scene.apply_collision_object(rm)

    def carried_path_clear(self, traj: Any) -> bool:
        """True if no waypoint of ``traj`` puts the robot *or the attached item* in collision.

        Waypoints are replayed through the scene's own current state (the one
        that holds the attached item); the state's joints are restored after.
        """
        with self.psm.read_write() as scene:
            cur = scene.current_state
            saved = np.asarray(cur.get_joint_group_positions("panda_arm")).copy()
            try:
                for i in range(len(traj)):
                    cur.set_joint_group_positions(
                        "panda_arm", np.asarray(traj[i].get_joint_group_positions("panda_arm"))
                    )
                    cur.update()
                    if scene.is_state_colliding(robot_state=cur, joint_model_group_name="panda_arm"):
                        return False
                return True
            finally:
                cur.set_joint_group_positions("panda_arm", saved)
                cur.update()

    def _remove_object(self, item_id: str) -> None:
        rm = CollisionObject(id=item_id, operation=CollisionObject.REMOVE)
        rm.header.frame_id = FRAME
        with self.psm.read_write() as scene:
            scene.apply_collision_object(rm)

    # --- motion pieces ------------------------------------------------------------
    def _ik(
        self, xyz: tuple[float, float, float], quat: tuple[float, float, float, float], seed: RobotState
    ) -> RobotState | None:
        # Seeds: the given state first (smooth, small motions), then 'ready' with joint1
        # turned toward the target: from a far-away seed (e.g. above a tray) the solver
        # sometimes found nothing for an easy pose (M3 batch: 2 items skipped).
        toward = np.array(READY_POSE)
        toward[0] = math.atan2(xyz[1], xyz[0])
        for q0 in (np.asarray(seed.get_joint_group_positions("panda_arm")), toward):
            st = RobotState(self.model)
            st.set_to_default_values()
            st.set_joint_group_positions("panda_arm", q0)
            st.update()
            if not st.set_from_ik("panda_arm", _pose(xyz, quat).pose, HAND, 0.2):
                continue
            st.update()
            # IK may return a joint exactly at its limit; MoveIt then rejects that state as a
            # plan start ("outside bounds"), and it leaves no room to move. Keep a margin.
            if not within_limits(st.get_joint_group_positions("panda_arm"), self.pc.joint_limit_margin):
                continue
            with self.psm.read_only() as scene:
                if scene.is_state_colliding(robot_state=st, joint_model_group_name="panda_arm"):
                    continue
            return st
        return None

    def _plan(self, start: RobotState, goal: RobotState | PoseStamped, kind: str) -> tuple[Any, str]:
        """Plan one arm segment; returns (trajectory, planner) or (None, '')."""
        attempts = {
            # Straight line; slower if joint limits object; last resort a short PTP to
            # the same pose (a few mm off the line over 10 cm). Labels are what the
            # segment records, so the fallback rate can be counted.
            "LIN": [("LIN", self.lin), ("LIN_slow", self.lin_slow), ("PTP", self.ptp)],
            "PTP": [("PTP", self.ptp), ("OMPL", self.ompl)],
            "CARRY": [("PTP", self.ptp_carry), ("OMPL", self.ompl)],  # holding an item
        }[kind]
        for name, params in attempts:
            self.arm.set_start_state(robot_state=start)
            if isinstance(goal, RobotState):
                self.arm.set_goal_state(robot_state=goal)
            else:
                self.arm.set_goal_state(pose_stamped_msg=goal, pose_link=HAND)
            result = self.arm.plan(single_plan_parameters=params)
            if result:
                return result.trajectory, name
        return None, ""

    @staticmethod
    def _end(traj: Any) -> RobotState:
        st = traj[len(traj) - 1]
        st.update()
        return st

    # --- the chain ---------------------------------------------------------------------
    def plan(self, it: Item, items: list[Item], box: Box, slot: int, start: RobotState) -> PickPlacePlan:
        """Plan the full chain for ``it`` (``items``: everything still on the table, incl. ``it``).

        Raises:
            PickPlaceError: with a reason code when the item can't be done.
        """
        reach = math.hypot(it.x, it.y)
        if reach > self.pc.reach_max:
            raise PickPlaceError("outside_workspace", f"{reach:.2f} m from the base > {self.pc.reach_max} m")
        obstacles = obstacles_for(it.id, items, self.cfg)
        clear = [
            g for g in candidates(it, self.cfg) if finger_clearance(g, obstacles) >= self.pc.clearance_min
        ]
        if not clear:
            raise PickPlaceError("no_collision_free_grasp", "every finger placement touches a neighbour")

        self._remove_object(it.id)  # the fingers must be allowed to touch it
        reasons: set[str] = set()
        stages: Counter[str] = Counter()  # where each candidate failed, for the error detail
        for g in clear:
            q = g.hand_quaternion()
            pre = self._ik(g.hand_position(self.pc.approach), q, start)
            grasp = self._ik(g.hand_position(0.0), q, pre) if pre is not None else None
            if pre is None or grasp is None:
                reasons.add("no_ik")
                stages["ik_pregrasp" if pre is None else "ik_grasp"] += 1
                continue
            plan = PickPlacePlan(it.id, box.label, g)
            failed = self._plan_pick(plan, start, pre)
            if failed:
                reasons.add("plan_failed")
                stages[failed] += 1
                continue
            self.attach(it)
            try:
                failed = self._plan_place(plan, box, slot)
            finally:
                self.detach()
            if not failed:
                return plan  # the item stays out of the world: it will be in the tray
            reasons.add("place_unreachable")
            stages[failed] += 1
        with self.psm.read_write() as scene:  # not done: back into the world for the other items
            scene.apply_collision_object(_item_object(it, self.cfg))
        # Report the furthest stage reached: place beats planning beats IK.
        code = next(c for c in ("place_unreachable", "plan_failed", "no_ik") if c in reasons)
        detail = ", ".join(f"{k} x{v}" for k, v in stages.most_common())
        raise PickPlaceError(code, f"{len(clear)} clear grasp candidates; failed at: {detail}")

    def _plan_pick(self, plan: PickPlacePlan, start: RobotState, pre: RobotState) -> str:
        """Plan the pick segments into ``plan``; returns '' or the stage that failed."""
        g = plan.grasp
        t1, n1 = self._plan(start, pre, "PTP")
        if t1 is None:
            return "to_pregrasp"
        t2, n2 = self._plan(self._end(t1), _pose(g.hand_position(0.0), g.hand_quaternion()), "LIN")
        if t2 is None:
            return "approach"
        t3, n3 = self._plan(
            self._end(t2), _pose(g.hand_position(self.pc.approach), g.hand_quaternion()), "LIN"
        )
        if t3 is None:
            return "lift"
        plan.segments += [
            Segment("open", gripper=g.opening / 2),
            Segment("to_pregrasp", t1, planner=n1),
            Segment("approach", t2, planner=n2),
            Segment("close", gripper=0.0),
            Segment("lift", t3, planner=n3),
        ]
        plan.end_state = self._end(t3)
        return ""

    def _plan_place(self, plan: PickPlacePlan, box: Box, slot: int) -> str:
        """Plan the place segments into ``plan``; returns '' or the stage that failed last."""
        dx, dy = self.pc.slot_offsets[slot % len(self.pc.slot_offsets)]
        x, y = box.x + dx, box.y + dy
        # Release with the fingertips above the walls: the item drops ~5 cm into the
        # tray, so the hand never enters it and any wrist angle works.
        z_tcp = self.cfg.surface_z + box.wall_height + self.pc.release_clearance + FINGERTIP_BELOW_TCP
        neutral = neutral_hand_yaw(x, y)  # least wrist rotation with joint1 pointing at the slot
        start = plan.end_state
        failed = "place_ik"
        for yaw_offset in (0.0, math.pi / 2, -math.pi / 2, math.pi):
            q = hand_down_quaternion(neutral + yaw_offset)
            above = (x, y, z_tcp + TCP_OFFSET + self.pc.approach)
            release = (x, y, z_tcp + TCP_OFFSET)
            goal_above = self._ik(above, q, start)
            if goal_above is None:
                continue
            t1, n1 = self._plan(start, goal_above, "CARRY")
            if t1 is None:
                failed = "to_box"
                continue
            # The item hangs below the fingers: check where it goes (the lift isn't
            # checked: it starts with the item resting on the table and goes straight up).
            if not self.carried_path_clear(t1):
                failed = "to_box_item_collision"
                continue
            t2, n2 = self._plan(self._end(t1), _pose(release, q), "LIN")
            if t2 is None:
                failed = "lower"
                continue
            if not self.carried_path_clear(t2):
                failed = "lower_item_collision"
                continue
            t3, n3 = self._plan(self._end(t2), _pose(above, q), "LIN")
            if t3 is None:
                failed = "retreat"
                continue
            plan.segments += [
                Segment("to_box", t1, planner=n1),
                Segment("lower", t2, planner=n2),
                Segment("release", gripper=plan.grasp.opening / 2),
                Segment("retreat", t3, planner=n3),
            ]
            plan.end_state = self._end(t3)
            return ""
        return failed
