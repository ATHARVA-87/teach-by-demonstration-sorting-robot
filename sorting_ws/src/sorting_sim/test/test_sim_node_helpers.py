"""Sim node helpers that don't need a ROS graph: camera model, yaw, time stamps."""

import math
import os

import mujoco
import numpy as np
import pytest
from builtin_interfaces.msg import Time

from sorting_sim.cell import CameraConfig, add_overhead_camera
from sorting_sim.sim_node import _stamp, camera_info, yaw_from_quaternion

os.environ.setdefault("MUJOCO_GL", "egl")


def test_stamp_splits_seconds() -> None:
    t = _stamp(12.25)
    assert (t.sec, t.nanosec) == (12, 250_000_000)


def test_yaw_from_quaternion_wraps_and_handles_tilt() -> None:
    for yaw in (0.0, 1.0, 3.0, -3.0, math.pi):
        q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        assert abs(math.remainder(yaw_from_quaternion(*q) - yaw, 2 * math.pi)) < 1e-9
    # 90 deg about x (item on its side), then yaw 0.5 about world z: heading is still 0.5
    qx = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    qz = (math.cos(0.25), 0.0, 0.0, math.sin(0.25))
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qx
    q = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    assert abs(yaw_from_quaternion(*q) - 0.5) < 1e-9


def test_camera_info_matches_rendered_image() -> None:
    """Flat markers at known table points must appear where CameraInfo projects them (< 0.25 px).

    Checks focal length, principal point convention ((W-1)/2) and axis
    directions (image right = world +y, image down = world +x).
    """
    cam = CameraConfig()
    spec = mujoco.MjSpec()
    add_overhead_camera(spec, cam)
    points = [(cam.center[0], cam.center[1]), (0.30, -0.20), (0.62, 0.25), (0.40, 0.30)]
    # Self-lit white squares on black, 1 mm tall (no visible sides). Their
    # anti-aliased edges give a sub-pixel intensity centroid; a binary mask
    # would quantize to +-0.5 px, the size of the convention error we test for.
    spec.visual.headlight.ambient = [0, 0, 0]
    spec.visual.headlight.diffuse = [0, 0, 0]
    spec.visual.quality.offsamples = 8
    spec.add_material(name="glow", rgba=[1, 1, 1, 1], emission=1.0)
    for x, y in points:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, 0.0005], size=[0.006, 0.006, 0.0005], material="glow"
        )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, cam.height_px, cam.width)
    except Exception as exc:  # no GPU / EGL on this machine
        pytest.skip(f"no off-screen rendering: {exc}")
    renderer.update_scene(data, camera="overhead")
    gray = renderer.render().astype(float).mean(axis=2)
    renderer.close()
    k_mat = np.array(camera_info(cam, Time()).k).reshape(3, 3)
    vv, uu = np.mgrid[0 : cam.height_px, 0 : cam.width]
    depth = cam.height - 0.001
    for k, (x, y) in enumerate(points):
        u = k_mat[0, 2] + k_mat[0, 0] * (y - cam.center[1]) / depth
        v = k_mat[1, 2] + k_mat[1, 1] * (x - cam.center[0]) / depth
        win = (np.abs(uu - u) < 12) & (np.abs(vv - v) < 12)  # around the expected spot
        w = gray * win
        assert w.sum() > 0, f"marker {k} not visible near ({u:.1f}, {v:.1f})"
        mu, mv = (w * uu).sum() / w.sum(), (w * vv).sum() / w.sum()
        assert abs(mu - u) < 0.2 and abs(mv - v) < 0.2, (k, mu, u, mv, v)
