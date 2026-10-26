"""Top-down grasp candidates for the sorting items (decision D8 #5). Pure geometry, no ROS.

A grasp is described by where the tool center point (TCP, between the finger
pads) goes and by the **closing angle**: the world-frame direction along which
the fingers close. Candidates come from each shape's symmetry:

- cube: across either pair of parallel faces (2 axes);
- cylinder: any axis (12 evenly spaced);
- triangular prism: no parallel faces, so across a flat side and the opposite
  edge (3 axes), with the TCP centered between them, not on the centroid.

Each axis gives two hand orientations (the hand rotated by 180 deg closes along
the same line); both are kept, IK and joint limits decide. Candidates are
ranked by how little the wrist must turn from its neutral orientation, and
filtered by **finger clearance**: both fingers, opened to the item width plus
a margin, must not touch a neighbouring item on the way down.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sorting_scene import Item, SceneConfig

# Panda hand geometry (Menagerie fingertip pads: 17 mm wide, centered 0.1034 m
# below the panda_hand origin). Finger extents are conservative: they include
# the finger body above the pad.
TCP_OFFSET = 0.1034  # panda_hand origin -> TCP along the approach axis
MAX_OPENING = 0.08  # total opening of the Panda hand
FINGER_THICKNESS = 0.018  # along the closing axis, outwards from the inner pad face
FINGER_WIDTH = 0.022  # across the closing axis
OPEN_MARGIN = 0.02  # opening = item width + margin (D8: 15-20 mm)
PRISM_CIRCUMRADIUS = 0.575  # x item size (same as the AR view and the sim)

# With the hand pointing down, orientation (w, x, y, z) = (0, 0.924, -0.383, 0)
# closes the fingers along world -135 deg (hand y axis). Rotating the hand by
# ``hand_yaw`` about world z turns the closing direction by the same angle.
_DOWN = (0.0, math.cos(math.pi / 8), -math.sin(math.pi / 8), 0.0)  # (0, 0.924, -0.383, 0), exact
_DOWN_CLOSING_ANGLE = -3 * math.pi / 4


def wrap(a: float) -> float:
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class Grasp:
    """One top-down grasp candidate."""

    item_id: str
    x: float  # TCP position (m, world)
    y: float
    z: float
    closing_angle: float  # world direction the fingers close along (rad)
    width: float  # item width between the pads (m)
    opening: float  # total opening commanded before the descent (m)

    @property
    def hand_yaw(self) -> float:
        """Rotation about world z applied to the hand-down orientation."""
        return wrap(self.closing_angle - _DOWN_CLOSING_ANGLE)

    def hand_quaternion(self) -> tuple[float, float, float, float]:
        """panda_hand orientation (w, x, y, z)."""
        return hand_down_quaternion(self.hand_yaw)

    def hand_position(self, height_above_tcp: float = 0.0) -> tuple[float, float, float]:
        """panda_hand origin for this grasp, optionally lifted (pre-grasp / retreat)."""
        return (self.x, self.y, self.z + TCP_OFFSET + height_above_tcp)


def hand_down_quaternion(hand_yaw: float) -> tuple[float, float, float, float]:
    """panda_hand pointing straight down, turned by ``hand_yaw`` about world z (w, x, y, z)."""
    h = hand_yaw / 2
    return _qmul((math.cos(h), 0.0, 0.0, math.sin(h)), _DOWN)


def _qmul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def neutral_hand_yaw(x: float, y: float) -> float:
    """Hand yaw of MoveIt's 'ready' wrist with joint1 turned toward (x, y).

    At 'ready' with joint1 = 0 the fingers close along -90 deg, i.e. hand yaw
    = +45 deg (closing = yaw - 135 deg); joint1 adds its angle. So the neutral
    hand yaw is atan2(y, x) + 45 deg (checked by FK in the planning test).
    """
    return wrap(math.atan2(y, x) + math.pi / 4)


def item_width(shape: str, size: float) -> float:
    """Distance between the finger pads when holding the item (m)."""
    if shape in ("cube", "cylinder"):
        return size
    if shape == "prism":
        return 1.5 * PRISM_CIRCUMRADIUS * size  # flat side to opposite edge
    raise ValueError(f"Unknown shape '{shape}'")


def footprint_radius(shape: str, size: float) -> float:
    """Radius of a circle around the item's footprint (conservative)."""
    if shape == "cube":
        return size * math.sqrt(2) / 2
    if shape == "cylinder":
        return size / 2
    if shape == "prism":
        return PRISM_CIRCUMRADIUS * size
    raise ValueError(f"Unknown shape '{shape}'")


def candidates(item: Item, cfg: SceneConfig, n_cylinder: int = 12) -> list[Grasp]:
    """All grasp candidates for ``item``, best first (least wrist rotation).

    Grasps are at the item's mid-height. Items wider than the hand can open
    get no candidates.
    """
    size = cfg.sizes[item.size]
    width = item_width(item.shape, size)
    opening = min(width + OPEN_MARGIN, MAX_OPENING)
    if width >= MAX_OPENING - 0.004:
        return []
    z = cfg.surface_z + cfg.item_height / 2
    axes: list[tuple[float, float]] = []  # (closing angle, TCP offset along it)
    if item.shape == "cube":
        axes = [(item.yaw, 0.0), (item.yaw + math.pi / 2, 0.0)]
    elif item.shape == "cylinder":
        axes = [(k * math.pi / n_cylinder, 0.0) for k in range(n_cylinder)]
    elif item.shape == "prism":
        # Along item-frame +y (rotated by k*120 deg) the prism spans from the
        # apex at -r to the flat side at +r/2: center the TCP at -r/4.
        r = PRISM_CIRCUMRADIUS * size
        axes = [(item.yaw + math.pi / 2 + k * 2 * math.pi / 3, -r / 4) for k in range(3)]
    grasps = []
    for angle, offset in axes:
        cx = item.x + offset * math.cos(angle)
        cy = item.y + offset * math.sin(angle)
        for flip in (0.0, math.pi):  # same line, hand turned around
            grasps.append(Grasp(item.id, cx, cy, z, wrap(angle + flip), width, opening))
    # Prefer the least wrist rotation from the neutral wrist (see neutral_hand_yaw).
    neutral = neutral_hand_yaw(item.x, item.y)
    grasps.sort(key=lambda g: abs(wrap(g.hand_yaw - neutral)))
    return grasps


def finger_rectangles(g: Grasp) -> list[tuple[float, float, float, float, float]]:
    """Footprints of the two open fingers: (cx, cy, angle, half_len, half_width).

    ``half_len`` runs along the closing axis, ``half_width`` across it.
    """
    d = g.opening / 2 + FINGER_THICKNESS / 2
    c, s = math.cos(g.closing_angle), math.sin(g.closing_angle)
    return [
        (g.x + sign * d * c, g.y + sign * d * s, g.closing_angle, FINGER_THICKNESS / 2, FINGER_WIDTH / 2)
        for sign in (1.0, -1.0)
    ]


def _rect_circle_distance(rect: tuple[float, float, float, float, float], px: float, py: float) -> float:
    cx, cy, angle, hl, hw = rect
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = px - cx, py - cy
    u, v = dx * c + dy * s, -dx * s + dy * c  # point in the rectangle frame
    ou, ov = max(abs(u) - hl, 0.0), max(abs(v) - hw, 0.0)
    return math.hypot(ou, ov)


def finger_clearance(g: Grasp, obstacles: Iterable[tuple[float, float, float]]) -> float:
    """Smallest gap between an open finger and an obstacle circle (x, y, radius); inf if none."""
    gap = math.inf
    for rect in finger_rectangles(g):
        for ox, oy, radius in obstacles:
            gap = min(gap, _rect_circle_distance(rect, ox, oy) - radius)
    return gap


def obstacles_for(item_id: str, items: Iterable[Item], cfg: SceneConfig) -> list[tuple[float, float, float]]:
    """Footprint circles of every item except ``item_id``."""
    return [(it.x, it.y, footprint_radius(it.shape, cfg.sizes[it.size])) for it in items if it.id != item_id]
