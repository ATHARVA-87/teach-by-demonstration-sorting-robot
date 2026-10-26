"""Build the simulated sorting cell as a MuJoCo model, one scene at a time.

Decision D8: the cell is rebuilt per scene with ``MjSpec`` (number of items,
their shape/size/color and, from M4, lighting and textures all change between
episodes, which a fixed-at-startup model can't do).

Contents:
- Franka Panda from MuJoCo Menagerie, loaded unmodified. Its internal names
  (``joint1..7``, ``finger_joint1/2``, ``hand``) are kept, because Menagerie's
  contact exclusions and gripper tendon refer to them; the sim node maps them
  to MoveIt's names (``panda_joint1..7``, ...) at the ROS interface
  (``JOINT_NAME_MAP``). Kinematics match the MoveIt URDF
  (test/test_fk_consistency.py).
- Table top at z = 0 (the robot is mounted on the table, world = robot base).
- Open trays for the boxes in ``scene.yaml``, each with its letter and fill
  pattern on the floor (design plan D6 rev. 11: same as the AR view).
- Items as free bodies: cube (box), cylinder, triangular prism (mesh), with
  the scene's paint colors.
- An overhead RGB camera ("overhead") looking straight down at the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np

from sorting_scene import Item, Scene, SceneConfig

# MuJoCo (Menagerie) joint name -> ROS / MoveIt joint name.
JOINT_NAME_MAP = {f"joint{i}": f"panda_joint{i}" for i in range(1, 8)} | {
    "finger_joint1": "panda_finger_joint1",
    "finger_joint2": "panda_finger_joint2",
}
PATTERNS = ("hatch", "dots", "plain", "crosshatch", "vertical")  # same order as ar_teach.render

# Arm pose that keeps the robot out of the overhead camera's view: MoveIt's
# 'ready' pose turned to face backwards (joint1 = 2.6 rad). Taken before every
# perception snapshot, as on a real cell. Verified: no arm pixels in a
# segmentation render (only the fixed base at the image edge), and free of
# self-collision in MoveIt's collision model with margin (a tighter folded pose
# also hid the arm but sits close to MoveIt's link5/link7 contact).
PARK_POSE = (2.6, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


@dataclass(frozen=True)
class CameraConfig:
    """Overhead camera: position above the table center, looking straight down."""

    center: tuple[float, float] = (0.48, 0.06)
    height: float = 1.0  # m above the table top
    fovy_deg: float = 45.0  # vertical field of view
    width: int = 640
    height_px: int = 480


@dataclass(frozen=True)
class PhysicsConfig:
    """Contact and mass settings for items."""

    item_density: float = 400.0  # kg/m^3 (light plastic/wood): 32 mm cube ≈ 13 g
    friction: tuple[float, float, float] = (1.2, 0.02, 0.002)  # sliding, torsional, rolling
    finger_friction: float = 1.5
    condim: int = 4  # 4 = sliding + torsional friction (stable pinch grasps)
    # Menagerie's gripper is a 100 N/m position servo (~1.4 N squeeze on a 28 mm
    # item; a real Franka hand: up to 70 N). x3 grips firmly; at x10 a prism's
    # edge sank ~13 mm into the soft finger-pad contact (sweep in M3).
    gripper_stiffness_scale: float = 3.0


@dataclass
class Cell:
    """A compiled cell and the names needed to read and drive it."""

    model: mujoco.MjModel
    item_ids: dict[str, int] = field(default_factory=dict)  # item id -> body id
    item_qpos: dict[str, int] = field(default_factory=dict)  # item id -> qpos address of its freejoint
    arm_joints: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))  # MuJoCo names
    finger_joints: tuple[str, ...] = ("finger_joint1", "finger_joint2")
    arm_actuators: tuple[str, ...] = tuple(f"actuator{i}" for i in range(1, 8))
    gripper_actuator: str = "actuator8"


def _tray_texture(label: str, slot: int, px: int = 128) -> np.ndarray:
    """RGB texture for a tray floor: anthracite + pattern + white letter (D6)."""
    img = np.full((px, px, 3), (66, 62, 56)[::-1], np.uint8)  # Anthracite #383E42 as RGB
    white = (236, 236, 231)
    pattern = PATTERNS[slot % len(PATTERNS)]
    step = 12
    if pattern in ("hatch", "crosshatch"):
        for k in range(-px, px, step):
            cv2.line(img, (k, px), (k + px, 0), white, 1, cv2.LINE_AA)
    if pattern == "crosshatch":
        for k in range(0, 2 * px, step):
            cv2.line(img, (k, 0), (k - px, px), white, 1, cv2.LINE_AA)
    if pattern == "dots":
        for y in range(step // 2, px, step):
            for x in range(step // 2, px, step):
                cv2.circle(img, (x, y), 1, white, -1, cv2.LINE_AA)
    if pattern == "vertical":
        for x in range(step // 2, px, step):
            cv2.line(img, (x, 0), (x, px), white, 1, cv2.LINE_AA)
    cv2.putText(img, label, (8, 34), cv2.FONT_HERSHEY_DUPLEX, 1.1, white, 2, cv2.LINE_AA)
    # MuJoCo maps texture rows along the geom's +y and columns along +x; the
    # overhead camera shows +y to the right and -x up, so rotate to read upright.
    return np.ascontiguousarray(np.rot90(img, k=1))


def _prism_vertices(size: float, height: float) -> list[float]:
    """Equilateral triangular prism, circumradius 0.575*size (as drawn in AR)."""
    r = 0.575 * size
    verts: list[float] = []
    for z in (-height / 2, height / 2):
        for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6):
            verts += [r * math.cos(a), r * math.sin(a), z]
    return verts


def build_cell(
    scene_cfg: SceneConfig,
    scene: Scene,
    panda_xml: str | Path,
    camera: CameraConfig | None = None,
    physics: PhysicsConfig | None = None,
) -> Cell:
    """Compile the cell for one scene.

    Raises:
        FileNotFoundError: if the Menagerie Panda model is missing (run
            ``bash scripts/fetch_robot_models.sh``).
    """
    panda_xml = Path(panda_xml)
    if not panda_xml.is_file():
        raise FileNotFoundError(
            f"Panda model not found at {panda_xml}. Run 'bash scripts/fetch_robot_models.sh'."
        )
    camera = camera or CameraConfig()
    physics = physics or PhysicsConfig()

    spec = mujoco.MjSpec.from_file(str(panda_xml))
    spec.modelname = "sorting_cell"
    # Gravity compensation on every robot link: the Franka controller cancels
    # gravity internally, and without it the PD position actuators sag (70 mrad
    # at the 'extended' pose, failing MoveIt's goal tolerance). Held items are
    # not compensated, like a real payload.
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    # Stiffer gripper servo (see PhysicsConfig.gripper_stiffness_scale).
    grip = spec.actuator("actuator8")
    grip.gainprm[0] *= physics.gripper_stiffness_scale
    grip.biasprm[1] *= physics.gripper_stiffness_scale
    grip.biasprm[2] *= physics.gripper_stiffness_scale
    # Fingers grip a little better than the default friction. Menagerie's
    # finger geoms are unnamed, so go through the finger bodies.
    for finger in ("left_finger", "right_finger"):
        for g in spec.body(finger).geoms:
            g.friction[0] = physics.finger_friction

    wb = spec.worldbody
    # Lights: one key light from above, soft ambient via the headlight.
    wb.add_light(
        name="key",
        pos=[0.45, 0.0, 1.6],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.45, 0.45, 0.45],
        specular=[0.05, 0.05, 0.05],
    )
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.15, 0.15, 0.15]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    # Table top: z in [-0.04, 0], large enough for the robot base and boxes.
    wb.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.30, 0.06, -0.02],
        size=[0.70, 0.62, 0.02],
        rgba=[0.55, 0.55, 0.53, 1.0],
    )

    # Boxes: open trays (floor + 4 walls), letter + pattern texture on the floor.
    wall = 0.004
    for b in scene_cfg.boxes:
        tex = spec.add_texture(
            name=f"tray_{b.label}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=128, height=128, nchannel=3
        )  # square, so rot90 keeps the size
        tex.data = _tray_texture(b.label, b.slot).tobytes()
        mat = spec.add_material(name=f"tray_{b.label}_mat")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        body = wb.add_body(name=f"box_{b.label}", pos=[b.x, b.y, 0.0])
        half = b.size / 2
        body.add_geom(
            name=f"box_{b.label}_floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, 0, 0.002],
            size=[half, half, 0.002],
            material=mat.name,
        )
        h = b.wall_height / 2
        dark = [0.22, 0.24, 0.26, 1.0]
        for k, (dx, dy, sx, sy) in enumerate(
            ((half, 0, wall, half), (-half, 0, wall, half), (0, half, half, wall), (0, -half, half, wall))
        ):
            body.add_geom(
                name=f"box_{b.label}_wall{k}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[dx, dy, h],
                size=[sx, sy, h],
                rgba=dark,
            )

    # Items: free bodies resting on the table.
    cell_items: list[Item] = list(scene.items)
    hz = scene_cfg.item_height / 2
    for it in cell_items:
        size = scene_cfg.sizes[it.size]
        rgba = [c / 255.0 for c in scene_cfg.colors[it.color]] + [1.0]
        body = wb.add_body(
            name=it.id, pos=[it.x, it.y, hz + 0.0005], quat=[math.cos(it.yaw / 2), 0, 0, math.sin(it.yaw / 2)]
        )
        body.add_freejoint(name=f"{it.id}_free")
        kw = {
            "rgba": rgba,
            "density": physics.item_density,
            "friction": list(physics.friction),
            "condim": physics.condim,
        }
        if it.shape == "cube":
            body.add_geom(
                name=f"{it.id}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[size / 2, size / 2, hz], **kw
            )
        elif it.shape == "cylinder":
            body.add_geom(
                name=f"{it.id}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[size / 2, hz, 0], **kw
            )
        elif it.shape == "prism":
            mesh = spec.add_mesh(name=f"{it.id}_mesh")
            mesh.uservert = _prism_vertices(size, scene_cfg.item_height)
            body.add_geom(name=f"{it.id}_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, **kw)
        else:
            raise ValueError(f"Unknown shape '{it.shape}'")

    add_overhead_camera(spec, camera)

    model = spec.compile()
    cell = Cell(model=model)
    for it in cell_items:
        cell.item_ids[it.id] = model.body(it.id).id
        cell.item_qpos[it.id] = int(model.jnt_qposadr[model.joint(f"{it.id}_free").id])
    return cell


def add_overhead_camera(spec: mujoco.MjSpec, camera: CameraConfig) -> None:
    """The overhead camera ("overhead"), looking straight down.

    Image right = world +y, image down = world +x (robot at the top), matching
    the AR view's orientation.
    """
    cx, cy = camera.center
    spec.worldbody.add_camera(
        name="overhead", pos=[cx, cy, camera.height], xyaxes=[0, 1, 0, -1, 0, 0], fovy=camera.fovy_deg
    )
    spec.visual.global_.offwidth = max(camera.width, spec.visual.global_.offwidth)
    spec.visual.global_.offheight = max(camera.height_px, spec.visual.global_.offheight)


def reset_to_home(cell: Cell, data: mujoco.MjData) -> None:
    """Robot to Menagerie's 'home' pose, items at their scene poses.

    Don't use ``mj_resetDataKeyframe`` directly: the keyframe was authored
    before the items were added, so its qpos for the item free joints is
    zero-padded and would teleport every item to the world origin.
    """
    m = cell.model
    mujoco.mj_resetData(m, data)  # qpos = qpos0: items at their spawn poses
    key = m.key("home").id
    n_robot = 9  # 7 arm + 2 finger joints come first in the Menagerie model
    data.qpos[:n_robot] = m.key_qpos[key][:n_robot]
    data.ctrl[:] = m.key_ctrl[key]
    mujoco.mj_forward(m, data)
