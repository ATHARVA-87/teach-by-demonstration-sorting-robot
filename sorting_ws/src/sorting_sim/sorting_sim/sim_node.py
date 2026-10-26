"""sorting_sim node: the simulated cell behind ros2_control-compatible interfaces.

Decision D8: same names and types as a real Panda driven by ros2_control, so
MoveIt and the executor can't tell the difference (and a ``mujoco_ros2_control``
backend or a real robot can be swapped in by launch file).

Topics out:
    /clock                       rosgraph_msgs/Clock           sim time (use_sim_time everywhere else)
    /joint_states                sensor_msgs/JointState        panda_joint1..7, panda_finger_joint1/2
    /overhead_camera/image_raw   sensor_msgs/Image             bgr8, rendered on its own thread (EGL)
    /overhead_camera/camera_info sensor_msgs/CameraInfo
    /sim/debug/item_poses        sorting_interfaces/ItemPoses  GROUND TRUTH: evaluation/debug only
Actions:
    /panda_arm_controller/follow_joint_trajectory  control_msgs/FollowJointTrajectory
    /panda_hand_controller/gripper_cmd             control_msgs/ParallelGripperCommand
Services:
    ~/reset_scene                sorting_interfaces/ResetScene new randomized scene

Action semantics (like joint_trajectory_controller / parallel_gripper_action_controller):
- A trajectory succeeds once the arm is within ``goal_tolerance_rad`` of the
  last point *and* stopped. A new trajectory goal preempts the running one,
  which is aborted ("preempted by a new goal"); a scene reset aborts both
  actions' goals ("scene reset during the goal"). Never reported as success.
  Those aborts carry ``INVALID_GOAL`` (FollowJointTrajectory has no code for
  "interrupted"); the ``error_string`` says what happened.
- Malformed trajectories are rejected up front (``INVALID_JOINTS``): no
  joints or points, unknown or repeated joints, non-increasing times, or any
  NaN/infinite value.
- Not implemented (documented, not silently honoured): per-goal path/goal
  tolerances in the request, a future ``header.stamp`` start time, and the
  gripper's ``max_effort`` (the hand's actuator force limit applies).
- Gripper: ``reached_goal`` when the finger is within ``gripper_tolerance_m``
  of the target, ``stalled`` when it stopped short (usually: holding an item).
  An empty ``command.position`` is rejected. On cancel the fingers hold where
  they are.

Threads: physics (paced to ``real_time_factor``), rendering (the EGL context
is bound to the thread that creates it), and the ROS executor.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Callable

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
from geometry_msgs.msg import Pose
from rclpy.action import ActionServer, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from sorting_interfaces.msg import Item as ItemMsg
from sorting_interfaces.msg import ItemPoses
from sorting_interfaces.srv import ResetScene
from std_msgs.msg import Header

from sorting_common.image_utils import IMAGE_QOS, cv2_to_imgmsg
from sorting_scene import default_config_path, generate_scene, load_config
from sorting_sim.cell import CameraConfig, PhysicsConfig
from sorting_sim.sim_core import CONTROL_STEPS, SimCore

# (predicate checked after each physics chunk, future to complete when it holds)
_Watcher = tuple[Callable[[], bool], Future[bool]]
CAMERA_FRAME = "overhead_camera"


def _stamp(t: float) -> Time:
    sec = int(t)
    return Time(sec=sec, nanosec=int((t - sec) * 1e9))


def camera_info(cam: CameraConfig, stamp: Time, frame_id: str = CAMERA_FRAME) -> CameraInfo:
    """Pinhole intrinsics of the MuJoCo camera (square pixels, no distortion).

    Principal point in the ROS/OpenCV convention (pixel centers at integer
    coordinates): the image center is at ((W-1)/2, (H-1)/2).
    """
    f = (cam.height_px / 2) / math.tan(math.radians(cam.fovy_deg) / 2)
    cx, cy = (cam.width - 1) / 2, (cam.height_px - 1) / 2
    info = CameraInfo()
    info.header.stamp, info.header.frame_id = stamp, frame_id
    info.width, info.height = cam.width, cam.height_px
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def yaw_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    """Heading about world z, in (-pi, pi] (valid for any orientation, not only upright)."""
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class SimNode(Node):
    """Runs the MuJoCo cell and exposes it like real hardware."""

    def __init__(self) -> None:
        super().__init__("sorting_sim")
        p = self.declare_parameter
        scene_yaml = str(p("scene_config", "").value) or str(default_config_path())
        panda = str(p("panda_xml", "").value) or os.environ.get("SORTING_PANDA_XML", "")
        self.rtf = float(p("real_time_factor", 1.0).value)  # <= 0: as fast as possible
        self.js_period = 1.0 / float(p("joint_state_hz", 100.0).value)  # sim time
        self.cam_period = 1.0 / float(p("camera_hz", 10.0).value)
        self.debug_period = 1.0 / float(p("debug_pose_hz", 10.0).value)
        self.goal_tolerance = float(p("goal_tolerance_rad", 0.02).value)
        self.stopped_velocity_tolerance = float(p("stopped_velocity_tolerance", 0.01).value)  # rad/s
        self.goal_time_tolerance = float(p("goal_time_tolerance_s", 1.0).value)
        self.gripper_tolerance = float(p("gripper_tolerance_m", 0.002).value)
        self.gripper_stall_velocity = float(p("gripper_stall_velocity", 1e-3).value)  # m/s
        self.gripper_stall_time = float(p("gripper_stall_time_s", 0.2).value)
        self.gripper_timeout = float(p("gripper_timeout_s", 3.0).value)
        seed = int(p("scene_seed", 0).value)
        n_items = int(p("n_items", 0).value) or None
        d_cam, d_phys = CameraConfig(), PhysicsConfig()
        self.cam = CameraConfig(
            center=(
                float(p("camera.center_x", d_cam.center[0]).value),
                float(p("camera.center_y", d_cam.center[1]).value),
            ),
            height=float(p("camera.height", d_cam.height).value),
            fovy_deg=float(p("camera.fovy_deg", d_cam.fovy_deg).value),
            width=int(p("camera.width", d_cam.width).value),
            height_px=int(p("camera.height_px", d_cam.height_px).value),
        )
        friction = [float(v) for v in p("physics.item_friction", list(d_phys.friction)).value]
        physics = PhysicsConfig(
            item_density=float(p("physics.item_density", d_phys.item_density).value),
            friction=(friction[0], friction[1], friction[2]),
            finger_friction=float(p("physics.finger_friction", d_phys.finger_friction).value),
            condim=int(p("physics.condim", d_phys.condim).value),
            gripper_stiffness_scale=float(
                p("physics.gripper_stiffness_scale", d_phys.gripper_stiffness_scale).value
            ),
        )

        self.scene_cfg = load_config(scene_yaml)
        self.sim = SimCore(
            self.scene_cfg, panda, generate_scene(self.scene_cfg, seed, n_items), self.cam, physics
        )

        cb = ReentrantCallbackGroup()
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.img_pub = self.create_publisher(Image, "/overhead_camera/image_raw", IMAGE_QOS)
        self.info_pub = self.create_publisher(CameraInfo, "/overhead_camera/camera_info", 10)
        self.debug_pub = self.create_publisher(ItemPoses, "/sim/debug/item_poses", 10)
        ActionServer(
            self,
            FollowJointTrajectory,
            "/panda_arm_controller/follow_joint_trajectory",
            self._on_trajectory,
            callback_group=cb,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
        )
        ActionServer(
            self,
            ParallelGripperCommand,
            "/panda_hand_controller/gripper_cmd",
            self._on_gripper,
            callback_group=cb,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
        )
        self.create_service(ResetScene, "~/reset_scene", self._on_reset, callback_group=cb)

        self._stop = threading.Event()
        self._throttle: dict[str, tuple[float, int]] = {}  # log key -> (last report, suppressed count)
        self._watch_lock = threading.Lock()
        self._watchers: list[_Watcher] = []
        self._render_ready = threading.Event()
        self._physics = threading.Thread(target=self._physics_loop, name="physics", daemon=True)
        self._render = threading.Thread(target=self._render_loop, name="render", daemon=True)
        # Renderer first: creating the EGL context stalls Python ~0.5 s, which
        # would otherwise show up as a physics hiccup right after startup.
        self._render.start()
        if not self._render_ready.wait(timeout=10.0):
            self.get_logger().warning("Camera renderer not ready after 10 s; starting physics anyway.")
        self._physics.start()
        self.get_logger().info(
            f"Simulated cell ready: scene seed {seed}, {len(self.sim.scene.items)} items, "
            f"real-time factor {self.rtf if self.rtf > 0 else 'max'}"
        )

    def _log_throttled(self, key: str, msg: str, period: float = 5.0) -> None:
        """Error log at most once per ``period`` s per ``key`` (with a count of the suppressed ones)."""
        now = time.monotonic()
        last, count = self._throttle.get(key, (-math.inf, 0))
        if now - last >= period:
            extra = f" ({count} more since the last report)" if count else ""
            self.get_logger().error(msg + extra)
            self._throttle[key] = (now, 0)
        else:
            self._throttle[key] = (last, count + 1)

    # --- physics thread -----------------------------------------------------------
    def _physics_loop(self) -> None:
        dt_chunk = CONTROL_STEPS * self.sim.timestep  # one control update (10 ms at dt = 2 ms)
        # Publish every k-th chunk (integer counts: float time comparisons skip ticks).
        js_every = max(1, round(self.js_period / dt_chunk))
        dbg_every = max(1, round(self.debug_period / dt_chunk))
        k = 0
        wall0, sim0 = time.monotonic(), self.sim.time
        last_warn = 0.0
        while not self._stop.is_set():
            try:
                self.sim.step(CONTROL_STEPS)
                k += 1
                t = self.sim.time
                stamp = _stamp(t)
                self.clock_pub.publish(Clock(clock=stamp))
                self._check_watchers()
                if k % js_every == 0:
                    names, pos, vel = self.sim.joint_state()
                    self.js_pub.publish(
                        JointState(name=names, position=pos, velocity=vel, header=self._hdr(stamp))
                    )
                if k % dbg_every == 0:
                    self._publish_debug(stamp)
                if self.rtf > 0:  # pace: sim time advances rtf x wall time
                    ahead = (t - sim0) / self.rtf - (time.monotonic() - wall0)
                    if ahead > 0:
                        time.sleep(ahead)
                    elif ahead < -0.5:  # can't keep up: re-anchor instead of racing to catch up
                        now = time.monotonic()
                        if now - last_warn > 10.0:
                            last_warn = now
                            self.get_logger().warning(
                                f"Simulation is {-ahead:.1f} s behind real time "
                                f"(real_time_factor {self.rtf}); is the CPU busy?"
                            )
                        wall0, sim0 = now, t
            except Exception as exc:  # never let physics die silently
                if not rclpy.ok():
                    return
                self._log_throttled("physics", f"Physics step failed: {exc}")
                time.sleep(0.1)

    @staticmethod
    def _hdr(stamp: Time) -> Header:
        return Header(stamp=stamp, frame_id="world")

    def _publish_debug(self, stamp: Time) -> None:
        msg = ItemPoses(header=self._hdr(stamp))
        for it, pos, quat in self.sim.items_with_poses():
            w, x, y, z = (float(v) for v in quat)
            msg.items.append(
                ItemMsg(
                    id=it.id,
                    color=it.color,
                    size=it.size,
                    shape=it.shape,
                    x=float(pos[0]),
                    y=float(pos[1]),
                    yaw=yaw_from_quaternion(w, x, y, z),
                )
            )
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in pos)
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = w, x, y, z
            msg.poses.append(pose)
        self.debug_pub.publish(msg)

    # --- render thread ---------------------------------------------------------------
    def _render_loop(self) -> None:
        renderer: mujoco.Renderer | None = None
        version = -1
        while not self._stop.is_set():
            t_wall = time.monotonic()
            try:
                # Version, model and state are taken together; the renderer is
                # rebuilt for every new model (MuJoCo would render data from a
                # different model without complaint, reading out of bounds).
                # Built even with no subscriber: creating the EGL context stalls
                # Python ~0.5 s, better at startup than mid-task.
                snap_version, model, data = self.sim.snapshot()
                if renderer is None or snap_version != version:
                    if renderer is not None:
                        renderer.close()
                    renderer = mujoco.Renderer(model, self.cam.height_px, self.cam.width)
                    version = snap_version
                    self._render_ready.set()
                if self.img_pub.get_subscription_count() > 0:
                    renderer.update_scene(data, camera="overhead")
                    rgb = renderer.render()
                    stamp = _stamp(float(data.time))
                    self.img_pub.publish(
                        cv2_to_imgmsg(np.ascontiguousarray(rgb[:, :, ::-1]), stamp, CAMERA_FRAME)
                    )
                    self.info_pub.publish(camera_info(self.cam, stamp))
                else:
                    time.sleep(0.1)
                    continue
            except Exception as exc:
                if not rclpy.ok():
                    break
                self._log_throttled(
                    "render", f"Rendering failed: {exc}. Check MUJOCO_GL=egl and the GPU driver."
                )
                self._render_ready.set()  # don't hold physics back
                time.sleep(1.0)
            # camera_hz is in sim time: at real-time factor r, a frame every period / r wall seconds.
            period = self.cam_period / self.rtf if self.rtf > 0 else self.cam_period
            time.sleep(max(0.0, period - (time.monotonic() - t_wall)))
        if renderer is not None:
            renderer.close()

    # --- actions ----------------------------------------------------------------------
    # The action callbacks are coroutines that await futures completed by the
    # physics thread (``_until``). A blocking callback that polls with sleep()
    # kept rclpy's executor busy and starved physics of the GIL (sim fell to
    # ~0.55x real time during every motion); awaiting costs nothing, and the
    # checks run on sim time, once per 10 ms control chunk.
    def _until(self, predicate: Callable[[], bool]) -> Future[bool]:
        """Future that completes once ``predicate()`` is true after a physics chunk."""
        fut: Future[bool] = Future()
        with self._watch_lock:
            self._watchers.append((predicate, fut))
        return fut

    def _check_watchers(self) -> None:
        with self._watch_lock:
            if not self._watchers:
                return
            fire: list[_Watcher] = []
            keep: list[_Watcher] = []
            for pred, fut in self._watchers:
                try:
                    done = pred()
                except Exception as exc:  # let the waiting callback re-check and report
                    self._log_throttled("watcher", f"Action check failed: {exc}")
                    done = True
                (fire if done else keep).append((pred, fut))
            self._watchers = keep
        for _, fut in fire:
            fut.set_result(True)

    def _interrupted(self, version: int, token_name: str, token: int) -> str:
        """Why a goal lost its world ('' if it didn't): scene reset or a newer goal."""
        if self.sim.version != version:
            return "scene reset during the goal"
        if getattr(self.sim, token_name) != token:
            return "preempted by a new goal"
        return ""

    async def _on_trajectory(self, goal_handle: ServerGoalHandle) -> FollowJointTrajectory.Result:
        traj = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()
        try:
            times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points]
            vel = (
                [list(p.velocities) for p in traj.points]
                if traj.points and all(p.velocities for p in traj.points)
                else None
            )
            version, tid = self.sim.start_trajectory(
                traj.joint_names, times, [list(p.positions) for p in traj.points], vel
            )
        except ValueError as exc:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = str(exc)
            self.get_logger().error(f"Rejected trajectory: {exc}")
            return result
        target = list(traj.points[-1].positions)
        names = list(traj.joint_names)

        def lost() -> str:
            return self._interrupted(version, "trajectory_id", tid)

        def settled() -> bool:
            # Like joint_trajectory_controller: within the goal tolerance *and*
            # stopped, so the state MoveIt reads next is the settled one.
            return (
                self.sim.arm_error(names, target) <= self.goal_tolerance
                and self.sim.arm_speed(names) <= self.stopped_velocity_tolerance
            )

        await self._until(
            lambda: bool(lost()) or goal_handle.is_cancel_requested or self.sim.trajectory_done()
        )
        if not lost() and not goal_handle.is_cancel_requested:
            deadline = self.sim.time + self.goal_time_tolerance
            await self._until(
                lambda: (
                    bool(lost()) or goal_handle.is_cancel_requested or settled() or self.sim.time >= deadline
                )
            )
        why = lost()
        if why:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = f"Trajectory aborted: {why}"
            self.get_logger().warning(result.error_string)
            return result
        if goal_handle.is_cancel_requested:
            self.sim.cancel_trajectory(tid)
            goal_handle.canceled()
            return result
        err = self.sim.arm_error(names, target)
        if not err <= self.goal_tolerance:  # NaN-safe: anything not provably within tolerance fails
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = f"Final joint error {err:.3f} rad > {self.goal_tolerance} rad"
            self.get_logger().warning(result.error_string)
            return result
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    async def _on_gripper(self, goal_handle: ServerGoalHandle) -> ParallelGripperCommand.Result:
        cmd = goal_handle.request.command
        result = ParallelGripperCommand.Result()
        if not cmd.position or not math.isfinite(cmd.position[0]):
            goal_handle.abort()
            self.get_logger().error(
                f"Rejected gripper goal: command.position needs one finite value, got {list(cmd.position)}"
            )
            return result
        target = float(cmd.position[0])
        version, gid = self.sim.set_gripper(target)
        start = self.sim.time
        still = {"since": start}  # sim time since the fingers stopped moving

        def finished() -> bool:
            if self._interrupted(version, "gripper_id", gid) or goal_handle.is_cancel_requested:
                return True
            pos, vel = self.sim.finger_state()
            now = self.sim.time
            if abs(vel) >= self.gripper_stall_velocity:
                still["since"] = now
            return (
                abs(pos - target) <= self.gripper_tolerance
                or now - still["since"] >= self.gripper_stall_time  # stopped short: holding something
                or now - start >= self.gripper_timeout
            )

        await self._until(finished)
        why = self._interrupted(version, "gripper_id", gid)
        if why:  # never report a grasp from a world that no longer exists
            goal_handle.abort()
            self.get_logger().warning(f"Gripper goal aborted: {why}")
            return result
        if goal_handle.is_cancel_requested:
            self.sim.hold_gripper(gid)
            goal_handle.canceled()
            return result
        pos, _ = self.sim.finger_state()
        result.reached_goal = abs(pos - target) <= self.gripper_tolerance
        result.stalled = not result.reached_goal
        names, positions, velocities = self.sim.joint_state()
        result.state = JointState(name=names[7:], position=positions[7:], velocity=velocities[7:])
        goal_handle.succeed()
        return result

    def _on_reset(self, req: ResetScene.Request, resp: ResetScene.Response) -> ResetScene.Response:
        try:
            scene = generate_scene(self.scene_cfg, int(req.seed), int(req.n_items) or None)
            self.sim.reset_scene(scene)
        except Exception as exc:  # report; the old scene keeps running unchanged
            resp.success, resp.message = False, f"Could not build scene {req.seed}: {exc}"
            self.get_logger().error(resp.message)
            return resp
        resp.success, resp.message = True, f"Scene {req.seed}: {len(scene.items)} items"
        self.get_logger().info(resp.message)
        return resp

    def destroy_node(self) -> None:
        """Stop the physics and render threads."""
        self._stop.set()
        self._physics.join(timeout=2.0)
        self._render.join(timeout=2.0)
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run sorting_sim sim_node`` (multi-threaded executor for actions)."""
    # Physics, rendering and four executor threads share the GIL; a shorter
    # switch interval (default 5 ms) keeps physics pacing smooth.
    sys.setswitchinterval(0.0005)
    rclpy.init(args=args)
    node = SimNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
