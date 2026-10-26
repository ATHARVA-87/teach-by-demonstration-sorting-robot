"""Grasp geometry (pure, no ROS)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sorting_motion.grasps import (
    FINGER_THICKNESS,
    MAX_OPENING,
    OPEN_MARGIN,
    PRISM_CIRCUMRADIUS,
    candidates,
    finger_clearance,
    item_width,
    neutral_hand_yaw,
    obstacles_for,
    wrap,
)

from sorting_scene import Item, default_config_path, load_config

CFG = load_config(default_config_path())


def _rot(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _same_line(a: float, b: float) -> bool:
    return abs(math.sin(a - b)) < 1e-9  # directions a and b span the same line


@pytest.mark.parametrize("yaw", [0.0, 0.4, -1.3, 3.0])
def test_hand_points_down_and_fingers_close_along_the_grasp_axis(yaw: float) -> None:
    """The bug this guards against: the first grasp test closed cubes corner to corner (45 deg off)."""
    it = Item("i", "red", "medium", "cube", 0.45, -0.1, yaw)
    for g in candidates(it, CFG):
        r = _rot(g.hand_quaternion())
        assert np.allclose(r[:, 2], [0, 0, -1], atol=1e-9)  # approach axis straight down
        closing = math.atan2(r[1, 1], r[0, 1])  # panda_hand y axis = finger travel
        assert _same_line(closing, g.closing_angle)


def test_cube_grasps_across_faces() -> None:
    it = Item("i", "red", "large", "cube", 0.45, -0.1, 0.3)
    gs = candidates(it, CFG)
    assert len(gs) == 4  # 2 face pairs x 2 hand orientations
    for g in gs:
        assert _same_line(g.closing_angle, 0.3) or _same_line(g.closing_angle, 0.3 + math.pi / 2)
        assert g.width == pytest.approx(CFG.sizes["large"])
        assert (g.x, g.y) == pytest.approx((0.45, -0.1))
        assert g.z == pytest.approx(CFG.surface_z + CFG.item_height / 2)  # mid-height


def test_prism_grasp_is_centered_between_flat_side_and_edge() -> None:
    it = Item("i", "red", "medium", "prism", 0.4, 0.0, 0.0)
    gs = candidates(it, CFG)
    assert len(gs) == 6  # 3 side/edge axes x 2 hand orientations
    r = PRISM_CIRCUMRADIUS * CFG.sizes["medium"]
    assert all(g.width == pytest.approx(1.5 * r) for g in gs)
    # Axis k=0 runs along +y: apex at -r, flat side at +r/2, so the TCP is at y = -r/4.
    g0 = next(g for g in gs if _same_line(g.closing_angle, math.pi / 2))
    assert (g0.x, g0.y) == pytest.approx((0.4, -r / 4))


def test_cylinder_has_evenly_spaced_axes() -> None:
    gs = candidates(Item("i", "red", "small", "cylinder", 0.4, 0.0, 0.0), CFG, n_cylinder=12)
    assert len(gs) == 24
    lines = sorted({round(wrap(g.closing_angle) % math.pi, 9) for g in gs})
    assert len(lines) == 12


def test_best_candidate_needs_least_wrist_turn() -> None:
    it = Item("i", "red", "small", "cube", 0.40, 0.25, 0.7)
    gs = candidates(it, CFG)
    neutral = neutral_hand_yaw(it.x, it.y)
    turns = [abs(wrap(g.hand_yaw - neutral)) for g in gs]
    assert turns == sorted(turns)


def test_opening_and_too_wide_items() -> None:
    assert item_width("cube", 0.04) + OPEN_MARGIN <= MAX_OPENING
    huge = load_config(default_config_path())
    object.__setattr__(huge, "sizes", {**huge.sizes, "large": 0.09})
    assert candidates(Item("i", "red", "large", "cube", 0.4, 0.0, 0.0), huge) == []


def test_finger_clearance_sees_neighbours() -> None:
    it = Item("a", "red", "small", "cube", 0.40, 0.0, 0.0)
    g = next(g for g in candidates(it, CFG) if _same_line(g.closing_angle, 0.0))  # fingers along x
    finger_center = g.opening / 2 + FINGER_THICKNESS / 2
    # A neighbour right where a finger goes: negative gap. Far away along y: clear.
    blocked = [Item("b", "red", "small", "cube", 0.40 + finger_center, 0.0, 0.0), it]
    clear = [Item("b", "red", "small", "cube", 0.40, 0.15, 0.0), it]
    assert finger_clearance(g, obstacles_for("a", blocked, CFG)) < 0
    assert finger_clearance(g, obstacles_for("a", clear, CFG)) > 0.05
    assert finger_clearance(g, []) == math.inf
