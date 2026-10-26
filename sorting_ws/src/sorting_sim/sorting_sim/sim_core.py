"""Simulation core (no ROS): physics stepping, trajectory following, gripper, state.

Thread-safety: every method that reads or changes the simulation state takes
``self.lock`` (``time``, ``timestep`` and ``arm_joint_names`` read single
values). A scene is built outside the lock and swapped in at once, so a
failed build changes nothing. The ROS node runs physics in one thread and
serves actions and rendering from others.

Goals must notice when the world changed under them: ``version`` increases
with every scene, ``trajectory_id`` with every trajectory and ``gripper_id``
with every gripper command.

Arm control: Menagerie's Panda has position actuators (actuator1..7, built-in
PD). A joint trajectory is followed by setting each actuator's target to the
trajectory interpolated at the current *sim* time (cubic Hermite when the
points carry velocities, else linear), exactly what ros2_control's
joint_trajectory_controller does with a position interface.

Gripper: actuator8 drives both fingers through a tendon; ctrl 0..255 maps
linearly to finger opening 0..0.04 m (each finger). The command reports
``reached_goal`` when the finger is within tolerance, and ``stalled`` when it
stopped short, typically because it is holding an item, like
``parallel_gripper_action_controller``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from sorting_scene import Item, Scene, SceneConfig
from sorting_sim.cell import (
    JOINT_NAME_MAP,
    PARK_POSE,
    CameraConfig,
    Cell,
    PhysicsConfig,
    build_cell,
    reset_to_home,
)

FINGER_MAX = 0.04  # m, opening of one finger (Panda hand: 8 cm total)
GRIPPER_CTRL_MAX = 255.0  # Menagerie actuator8 range
CONTROL_STEPS = 5  # physics steps per control update (100 Hz at dt = 2 ms)


@dataclass
class _Trajectory:
    joint_index: np.ndarray  # (J,) indices into the 7 arm joints
    times: np.ndarray  # (N,) seconds from start
    positions: np.ndarray  # (N, J)
    velocities: np.ndarray | None  # (N, J) or None
    t0: float  # sim time at start


def _hermite(t: float, tr: _Trajectory) -> np.ndarray:
    """Interpolate the trajectory at time ``t`` after its start."""
    ts = tr.times
    if t <= ts[0]:
        return tr.positions[0]
    if t >= ts[-1]:
        return tr.positions[-1]
    k = int(np.searchsorted(ts, t) - 1)
    h = ts[k + 1] - ts[k]
    s = (t - ts[k]) / h
    p0, p1 = tr.positions[k], tr.positions[k + 1]
    if tr.velocities is None:
        return p0 + (p1 - p0) * s
    v0, v1 = tr.velocities[k] * h, tr.velocities[k + 1] * h
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


class SimCore:
    """One simulated cell. Rebuilt per scene with :meth:`reset_scene`."""

    def __init__(
        self,
        scene_cfg: SceneConfig,
        panda_xml: str,
        scene: Scene,
        camera: CameraConfig | None = None,
        physics: PhysicsConfig | None = None,
    ) -> None:
        self.scene_cfg = scene_cfg
        self.panda_xml = panda_xml
        self.camera = camera or CameraConfig()
        self.physics = physics or PhysicsConfig()
        self.lock = threading.RLock()
        self._traj: _Trajectory | None = None
        self.version = 0
        self.trajectory_id = 0
        self.gripper_id = 0
        self.scene: Scene
        self.cell: Cell
        self.model: mujoco.MjModel
        self.data: mujoco.MjData
        self.reset_scene(scene)  # sets the four above

    # --- scene ------------------------------------------------------------------
    def reset_scene(self, scene: Scene, park: bool = True) -> None:
        """Build a new cell for ``scene`` and put the arm in the park pose.

        The new cell is built and settled without holding the lock (physics and
        /clock keep running meanwhile), then swapped in at once. If building
        fails, the exception propagates and the old scene stays untouched.
        """
        cell = build_cell(self.scene_cfg, scene, self.panda_xml, self.camera, self.physics)
        data = mujoco.MjData(cell.model)
        reset_to_home(cell, data)
        if park:
            data.qpos[:7] = PARK_POSE
            data.ctrl[:7] = PARK_POSE
        data.ctrl[7] = GRIPPER_CTRL_MAX
        mujoco.mj_forward(cell.model, data)
        mujoco.mj_step(cell.model, data, nstep=int(0.3 / cell.model.opt.timestep))  # items settle
        with self.lock:
            # Sim time keeps running across scenes: /clock must never jump back
            # (use_sim_time nodes such as MoveIt treat that as a time reset).
            data.time = float(self.data.time) if self.version else 0.0
            self.scene, self.cell, self.model, self.data = scene, cell, cell.model, data
            self._traj = None
            self.version += 1  # goals in progress and renderers notice this

    # --- stepping -----------------------------------------------------------------
    @property
    def time(self) -> float:
        """Simulation time in seconds."""
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        """Physics timestep in seconds."""
        return float(self.model.opt.timestep)

    def step(self, n: int = 1) -> None:
        """Advance physics ``n`` steps, following the active trajectory.

        Position targets are updated every ``CONTROL_STEPS`` physics steps
        (100 Hz at dt = 2 ms, the rate of the Panda's ros2_control config), and
        the steps in between run in one native call. One call per step hands
        Python's GIL back and forth hundreds of times a second, and with the
        ROS threads competing for it the sim fell well below real time.
        """
        with self.lock:
            done = 0
            while done < n:
                k = min(CONTROL_STEPS, n - done)
                if self._traj is not None:
                    q = _hermite(self.data.time - self._traj.t0, self._traj)
                    self.data.ctrl[self._traj.joint_index] = q
                mujoco.mj_step(self.model, self.data, nstep=k)
                done += k

    # --- arm ----------------------------------------------------------------------
    def arm_joint_names(self) -> list[str]:
        """ROS (MoveIt) names of the 7 arm joints, in actuator order."""
        return [JOINT_NAME_MAP[j] for j in self.cell.arm_joints]

    def start_trajectory(
        self,
        joint_names: Sequence[str],
        times: Sequence[float],
        positions: Sequence[Sequence[float]],
        velocities: Sequence[Sequence[float]] | None = None,
    ) -> tuple[int, int]:
        """Start following a joint trajectory (ROS joint names, times from now).

        Returns:
            (scene version, trajectory id), read together: a goal compares both
            later to notice a reset or a newer trajectory.

        Raises:
            ValueError: no joints or points, unknown or repeated joint names,
                non-increasing times, wrong point sizes, or a NaN/infinite value
                (a NaN target would otherwise drive the joint to 0 and the
                NaN error check could never fail).
        """
        names = self.arm_joint_names()
        if not joint_names:
            raise ValueError("Trajectory has no joint names")
        unknown = set(joint_names) - set(names)
        if unknown:
            raise ValueError(f"Unknown joints {sorted(unknown)}; this controller drives {names}")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"Repeated joint names in {list(joint_names)}")
        idx = np.array([names.index(j) for j in joint_names], dtype=int)
        t = np.asarray(times, float)
        p = np.asarray(positions, float)
        if t.size == 0:
            raise ValueError("Trajectory has no points")
        if p.ndim != 2 or p.shape != (len(t), len(idx)) or np.any(np.diff(t) <= 0) or t[0] < 0:
            raise ValueError(
                "Trajectory points need non-negative, strictly increasing times and one position per joint"
            )
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
            raise ValueError("Trajectory contains NaN or infinite times/positions")
        v = None
        if (
            velocities is not None
            and len(velocities) == len(t)
            and all(len(x) == len(idx) for x in velocities)
        ):
            v = np.asarray(velocities, float)
            if not np.all(np.isfinite(v)):
                raise ValueError("Trajectory contains NaN or infinite velocities")
        with self.lock:
            # A trajectory starting at t=0 begins from the *current* commanded
            # position, so prepend it when the first point is in the future.
            if t[0] > 0:
                t = np.concatenate([[0.0], t])
                p = np.vstack([self.data.ctrl[idx], p])
                if v is not None:
                    v = np.vstack([np.zeros(len(idx)), v])
            self._traj = _Trajectory(idx, t, p, v, self.data.time)
            self.trajectory_id += 1  # a goal still waiting on an older trajectory was preempted
            return self.version, self.trajectory_id

    def trajectory_done(self) -> bool:
        """True when the active trajectory's last point time has passed."""
        with self.lock:
            return self._traj is None or self.data.time - self._traj.t0 >= self._traj.times[-1]

    def cancel_trajectory(self, trajectory_id: int | None = None) -> None:
        """Hold the current position (stop following).

        With ``trajectory_id``, only if that trajectory is still the active one
        (a cancelled goal must not stop a newer goal's motion).
        """
        with self.lock:
            if trajectory_id is not None and trajectory_id != self.trajectory_id:
                return
            if self._traj is not None:
                self.data.ctrl[self._traj.joint_index] = self.data.qpos[self._traj.joint_index]
            self._traj = None

    def arm_error(self, joint_names: Sequence[str], target: Sequence[float]) -> float:
        """Max |actual - target| over the given joints (rad)."""
        names = self.arm_joint_names()
        with self.lock:
            q = np.array([self.data.qpos[names.index(j)] for j in joint_names])
        return float(np.max(np.abs(q - np.asarray(target))))

    def arm_speed(self, joint_names: Sequence[str]) -> float:
        """Max |velocity| over the given joints (rad/s)."""
        names = self.arm_joint_names()
        with self.lock:
            v = np.array([self.data.qvel[names.index(j)] for j in joint_names])  # arm dofs come first
        return float(np.max(np.abs(v)))

    # --- gripper -------------------------------------------------------------------
    def set_gripper(self, finger_position: float) -> tuple[int, int]:
        """Command one finger's opening (0 = closed, 0.04 = open), like the Panda hand.

        Returns:
            (scene version, gripper command id), read together.

        Raises:
            ValueError: NaN or infinite position.
        """
        if not math.isfinite(finger_position):
            raise ValueError(f"Gripper position must be finite, got {finger_position}")
        pos = float(np.clip(finger_position, 0.0, FINGER_MAX))
        with self.lock:
            self.data.ctrl[7] = pos / FINGER_MAX * GRIPPER_CTRL_MAX
            self.gripper_id += 1
            return self.version, self.gripper_id

    def hold_gripper(self, command_id: int | None = None) -> None:
        """Stop the fingers where they are (cancelled gripper goal).

        With ``command_id``, only if that command is still the current one.
        """
        with self.lock:
            if command_id is not None and command_id != self.gripper_id:
                return
            j = self.model.joint(self.cell.finger_joints[0])
            self.data.ctrl[7] = float(self.data.qpos[j.qposadr[0]]) / FINGER_MAX * GRIPPER_CTRL_MAX
            self.gripper_id += 1

    def finger_state(self) -> tuple[float, float]:
        """(finger_joint1 position, its velocity)."""
        with self.lock:
            j = self.model.joint(self.cell.finger_joints[0])
            return float(self.data.qpos[j.qposadr[0]]), float(self.data.qvel[j.dofadr[0]])

    # --- state for ROS -----------------------------------------------------------------
    def joint_state(self) -> tuple[list[str], list[float], list[float]]:
        """(ROS names, positions, velocities) of the 9 robot joints."""
        names, pos, vel = [], [], []
        with self.lock:
            for mj_name in (*self.cell.arm_joints, *self.cell.finger_joints):
                j = self.model.joint(mj_name)
                names.append(JOINT_NAME_MAP[mj_name])
                q = float(self.data.qpos[j.qposadr[0]])
                if mj_name in self.cell.finger_joints:
                    # Soft joint limits let the fingers overshoot 0.04 m by a hair;
                    # MoveIt rejects start states outside the URDF limits.
                    q = min(max(q, 0.0), FINGER_MAX)
                pos.append(q)
                vel.append(float(self.data.qvel[j.dofadr[0]]))
        return names, pos, vel

    def item_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Ground truth: item id -> (position xyz, quaternion wxyz). Evaluation/debug only."""
        with self.lock:
            return {
                iid: (self.data.xpos[bid].copy(), self.data.xquat[bid].copy())
                for iid, bid in self.cell.item_ids.items()
            }

    def snapshot(self) -> tuple[int, mujoco.MjModel, mujoco.MjData]:
        """(version, model, copy of the state), taken together (for rendering in another thread).

        A renderer must match the model: MuJoCo renders data from a different
        model without complaint, reading out of bounds.
        """
        with self.lock:
            d = mujoco.MjData(self.model)
            mujoco.mj_copyData(d, self.model, self.data)
            return self.version, self.model, d

    def items_with_poses(self) -> list[tuple[Item, np.ndarray, np.ndarray]]:
        """Ground truth: (scene item, position xyz, quaternion wxyz) for every item, read together."""
        with self.lock:
            return [
                (
                    it,
                    self.data.xpos[self.cell.item_ids[it.id]].copy(),
                    self.data.xquat[self.cell.item_ids[it.id]].copy(),
                )
                for it in self.scene.items
            ]
