"""D8 #4: the MuJoCo Panda (Menagerie) has the same kinematics as MoveIt's URDF.

MoveIt plans with ``moveit_resources_panda_description/urdf/panda.urdf``;
the simulator moves Menagerie's ``panda.xml``. Both are loaded into MuJoCo
(the URDF with its meshes stripped, fixed links kept) and compared over 100
seeded joint configurations: flange (``panda_link8``), hand, tool center point
and finger frames within 1 mm / 0.5 deg. Also checks that the two Menagerie
fingers stay coupled, like the URDF's mimic joint.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
PANDA = Path(
    os.environ.get("SORTING_PANDA_XML", ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml")
)
URDF = (
    Path("/opt/ros")
    / os.environ.get("ROS_DISTRO", "lyrical")
    / "share/moveit_resources_panda_description/urdf/panda.urdf"
)
pytestmark = pytest.mark.skipif(
    not (PANDA.is_file() and URDF.is_file()),
    reason="needs the Menagerie Panda and moveit_resources_panda_description",
)
TCP = np.array([0.0, 0.0, 0.1034])  # in the hand frame
ARM_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
ARM_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def _urdf_model() -> mujoco.MjModel:
    root = ET.parse(URDF).getroot()
    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    # Keep fixed links (panda_link8, panda_hand) as separate bodies. The URDF
    # has no inertias; kinematics don't need them, but MuJoCo requires mass.
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(
        mj, "compiler", fusestatic="false", discardvisual="true", boundmass="0.001", boundinertia="1e-6"
    )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _frame(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> tuple[np.ndarray, np.ndarray]:
    b = m.body(body).id
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def _angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    c = (np.trace(r1.T @ r2) - 1) / 2
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def test_menagerie_matches_moveit_urdf() -> None:
    mu, mm = _urdf_model(), mujoco.MjModel.from_xml_path(str(PANDA))
    du, dm = mujoco.MjData(mu), mujoco.MjData(mm)
    qadr_u = [mu.joint(f"panda_joint{i}").qposadr[0] for i in range(1, 8)]
    qadr_m = [mm.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
    fu = [mu.joint(f"panda_finger_joint{i}").qposadr[0] for i in (1, 2)]
    fm = [mm.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
    rng = np.random.default_rng(0)
    worst = {"pos_mm": 0.0, "rot_deg": 0.0}
    for _ in range(100):
        q = rng.uniform(ARM_LO, ARM_HI)
        f = float(rng.uniform(0.0, 0.04))
        du.qpos[qadr_u], dm.qpos[qadr_m] = q, q
        du.qpos[fu], dm.qpos[fm] = f, f
        mujoco.mj_kinematics(mu, du)
        mujoco.mj_kinematics(mm, dm)
        hand_p, hand_r = _frame(mm, dm, "hand")
        # Menagerie fuses link8 into the hand; URDF: hand = link8 rotated -45 deg about z.
        pairs = {
            "panda_link8": (hand_p, hand_r @ _rotz(math.pi / 4)),
            "panda_hand": (hand_p, hand_r),
            "panda_leftfinger": _frame(mm, dm, "left_finger"),
            # Same pose, different convention: Menagerie turns the right finger
            # *body* 180 deg about z, the URDF only turns its mesh.
            "panda_rightfinger": (
                _frame(mm, dm, "right_finger")[0],
                _frame(mm, dm, "right_finger")[1] @ _rotz(math.pi),
            ),
        }
        for urdf_body, (p_m, r_m) in pairs.items():
            p_u, r_u = _frame(mu, du, urdf_body)
            worst["pos_mm"] = max(worst["pos_mm"], float(np.linalg.norm(p_u - p_m)) * 1e3)
            worst["rot_deg"] = max(worst["rot_deg"], _angle_deg(r_u, r_m))
        p_u, r_u = _frame(mu, du, "panda_hand")
        tcp_err = np.linalg.norm((p_u + r_u @ TCP) - (hand_p + hand_r @ TCP)) * 1e3
        worst["pos_mm"] = max(worst["pos_mm"], float(tcp_err))
    assert worst["pos_mm"] < 1.0, worst
    assert worst["rot_deg"] < 0.5, worst


def test_fingers_stay_coupled() -> None:
    """The URDF's finger_joint2 mimics finger_joint1; Menagerie couples them with an equality constraint."""
    m = mujoco.MjModel.from_xml_path(str(PANDA))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    j1, j2 = m.joint("finger_joint1").qposadr[0], m.joint("finger_joint2").qposadr[0]
    for ctrl, expected in ((255.0, 0.04), (127.5, 0.02), (0.0, 0.0)):
        d.ctrl[7] = ctrl
        mujoco.mj_step(m, d, nstep=1000)
        assert abs(d.qpos[j1] - d.qpos[j2]) < 1e-4
        assert abs(d.qpos[j1] - expected) < 2e-3
