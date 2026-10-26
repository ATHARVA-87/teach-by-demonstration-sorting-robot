"""Scene model: the single definition of the sorting cell.

Loaded from ``config/scene.yaml`` and shared by the AR teaching view, the
simulator and the rule learner, so the taught scene and the robot's scene are
digital twins. Pure Python (no ROS), so it is importable everywhere and
unit-testable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ATTRIBUTE_NAMES: tuple[str, ...] = ("color", "size", "shape")


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in world XY (metres)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError(f"Invalid rectangle {self}")

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True if (x, y) lies inside, shrunk by ``margin`` on every side."""
        return (
            self.x_min + margin <= x <= self.x_max - margin
            and self.y_min + margin <= y <= self.y_max - margin
        )


@dataclass(frozen=True)
class Box:
    """A bin the robot sorts into. ``slot`` fixes its pattern in AR and MuJoCo."""

    label: str
    x: float
    y: float
    size: float
    wall_height: float
    slot: int

    def contains(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the box's square opening."""
        half = self.size / 2.0
        return abs(x - self.x) <= half and abs(y - self.y) <= half


@dataclass
class Item:
    """One item: attributes plus pose on the table (world frame)."""

    id: str
    color: str
    size: str
    shape: str
    x: float
    y: float
    yaw: float = 0.0

    def attributes(self) -> dict[str, str]:
        """The item's attribute values, keyed by attribute name."""
        return {"color": self.color, "size": self.size, "shape": self.shape}

    def describe(self) -> str:
        """Plain-language description, e.g. 'red cube small'."""
        return f"{self.color} {self.shape} {self.size}"


@dataclass
class Scene:
    """A concrete arrangement of items (the boxes come from the config)."""

    seed: int | None
    items: list[Item] = field(default_factory=list)

    def item(self, item_id: str) -> Item:
        """Look up an item by id (KeyError if absent)."""
        for it in self.items:
            if it.id == item_id:
                return it
        raise KeyError(item_id)

    def to_json(self) -> str:
        """Serialize to JSON (stable key order)."""
        return json.dumps({"seed": self.seed, "items": [asdict(i) for i in self.items]}, indent=2)

    @staticmethod
    def from_json(text: str) -> Scene:
        """Inverse of :meth:`to_json`."""
        data = json.loads(text)
        return Scene(seed=data.get("seed"), items=[Item(**i) for i in data["items"]])


@dataclass(frozen=True)
class ArView:
    """Exact affine map between world XY and AR-view pixels.

    ``p = c + s * R (w - w0)``, where the rows of R are the world directions
    of +u and +v. The axes must be orthonormal, so the map is a similarity
    (equal scale on both axes, which :meth:`meters_to_px` relies on). For the
    default config R = [[0, 1], [1, 0]]: an axis swap, i.e. a *reflection*
    (det = -1), which matches a mirrored webcam view. It round-trips exactly.
    """

    image_size: tuple[int, int]
    world_center: tuple[float, float]
    center_px: tuple[float, float]
    pixels_per_meter: float
    u_axis: tuple[float, float]
    v_axis: tuple[float, float]

    def __post_init__(self) -> None:
        if self.pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be positive")
        if abs(self._det()) < 1e-9:
            raise ValueError("u_axis and v_axis must not be parallel")
        nu, nv = math.hypot(*self.u_axis), math.hypot(*self.v_axis)
        dot = self.u_axis[0] * self.v_axis[0] + self.u_axis[1] * self.v_axis[1]
        if abs(nu - 1.0) > 1e-9 or abs(nv - 1.0) > 1e-9 or abs(dot) > 1e-9:
            raise ValueError("u_axis and v_axis must be orthonormal (unit length, perpendicular)")

    def _det(self) -> float:
        (a, b), (c, d) = self.u_axis, self.v_axis
        return a * d - b * c

    def world_to_px(self, x: float, y: float) -> tuple[float, float]:
        """World XY (m) -> pixel (u, v)."""
        dx, dy = x - self.world_center[0], y - self.world_center[1]
        s = self.pixels_per_meter
        u = self.center_px[0] + s * (self.u_axis[0] * dx + self.u_axis[1] * dy)
        v = self.center_px[1] + s * (self.v_axis[0] * dx + self.v_axis[1] * dy)
        return u, v

    def px_to_world(self, u: float, v: float) -> tuple[float, float]:
        """Pixel (u, v) -> world XY (m). Exact inverse of :meth:`world_to_px`."""
        s = self.pixels_per_meter
        pu, pv = (u - self.center_px[0]) / s, (v - self.center_px[1]) / s
        (a, b), (c, d) = self.u_axis, self.v_axis
        det = self._det()
        dx = (d * pu - b * pv) / det
        dy = (-c * pu + a * pv) / det
        return self.world_center[0] + dx, self.world_center[1] + dy

    def meters_to_px(self, meters: float) -> float:
        """Length in metres -> length in pixels (axes are orthonormal, so scale is equal)."""
        return meters * self.pixels_per_meter


@dataclass(frozen=True)
class SceneConfig:
    """Everything in ``scene.yaml``, validated."""

    surface_z: float
    items_area: Rect
    extent: Rect
    colors: dict[str, tuple[int, int, int]]
    sizes: dict[str, float]
    shapes: tuple[str, ...]
    item_height: float
    boxes: tuple[Box, ...]
    n_items_min: int
    n_items_max: int
    min_clearance: float
    max_tries: int
    ar_view: ArView

    @property
    def attribute_values(self) -> dict[str, tuple[str, ...]]:
        """Possible values per attribute, in config order."""
        return {"color": tuple(self.colors), "size": tuple(self.sizes), "shape": self.shapes}

    @property
    def box_labels(self) -> tuple[str, ...]:
        """Box labels in slot order."""
        return tuple(b.label for b in self.boxes)

    def box(self, label: str) -> Box:
        """Look up a box by label (KeyError if absent)."""
        for b in self.boxes:
            if b.label == label:
                return b
        raise KeyError(label)

    def box_at(self, x: float, y: float) -> Box | None:
        """The box whose opening contains (x, y), or None."""
        for b in self.boxes:
            if b.contains(x, y):
                return b
        return None

    def footprint_radius(self, size: str) -> float:
        """Radius of the circle enclosing an item footprint of this size."""
        # Cubes are the worst case: half-diagonal of the square.
        return self.sizes[size] * math.sqrt(2.0) / 2.0


def _req(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ValueError(f"scene config: missing '{key}' in {where}")
    return d[key]


def _pair(d: dict[str, Any], key: str) -> tuple[float, float]:
    value = _req(d, key, "ar_view")
    if len(value) != 2:
        raise ValueError(f"scene config: ar_view.{key} must have 2 numbers")
    return float(value[0]), float(value[1])


def load_config(path: str | Path) -> SceneConfig:
    """Load and validate ``scene.yaml``.

    Raises:
        ValueError: with a message naming the missing or invalid field.
    """
    raw = yaml.safe_load(Path(path).read_text())
    try:
        return _parse(raw)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            f"scene config {path}: missing or malformed field ({type(exc).__name__}: {exc})"
        ) from exc


def _parse(raw: dict[str, Any]) -> SceneConfig:
    table = _req(raw, "table", "root")
    attrs = _req(raw, "attributes", "root")
    gen = _req(raw, "generation", "root")
    ar = _req(raw, "ar_view", "root")

    colors: dict[str, tuple[int, int, int]] = {}
    for name, rgb in _req(attrs, "color", "attributes").items():
        if len(rgb) != 3 or not all(0 <= int(c) <= 255 for c in rgb):
            raise ValueError(f"scene config: color '{name}' must be 3 values in 0..255")
        colors[name] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    sizes = {k: float(v) for k, v in _req(attrs, "size", "attributes").items()}
    shapes = tuple(_req(attrs, "shape", "attributes"))
    if not colors or not sizes or not shapes:
        raise ValueError("scene config: every attribute needs at least one value")

    boxes = tuple(
        Box(
            label=str(_req(b, "label", "box")),
            x=float(b["x"]),
            y=float(b["y"]),
            size=float(b["size"]),
            wall_height=float(b["wall_height"]),
            slot=i,
        )
        for i, b in enumerate(_req(raw, "boxes", "root"))
    )
    labels = [b.label for b in boxes]
    if len(set(labels)) != len(labels) or len(boxes) < 2:
        raise ValueError("scene config: need at least 2 boxes with unique labels")

    items_area = Rect(**_req(table, "items_area", "table"))
    for b in boxes:
        if items_area.contains(b.x, b.y, margin=-b.size / 2):
            raise ValueError(f"scene config: box {b.label} overlaps the items area")

    n_items = _req(gen, "n_items", "generation")
    cfg = SceneConfig(
        surface_z=float(_req(table, "surface_z", "table")),
        items_area=items_area,
        extent=Rect(**_req(table, "extent", "table")),
        colors=colors,
        sizes=sizes,
        shapes=shapes,
        item_height=float(_req(attrs, "item_height", "attributes")),
        boxes=boxes,
        n_items_min=int(n_items["min"]),
        n_items_max=int(n_items["max"]),
        min_clearance=float(_req(gen, "min_clearance", "generation")),
        max_tries=int(_req(gen, "max_tries", "generation")),
        ar_view=ArView(
            image_size=(int(ar["image_size"][0]), int(ar["image_size"][1])),
            world_center=_pair(ar, "world_center"),
            center_px=_pair(ar, "center_px"),
            pixels_per_meter=float(ar["pixels_per_meter"]),
            u_axis=_pair(ar, "u_axis"),
            v_axis=_pair(ar, "v_axis"),
        ),
    )
    if not 1 <= cfg.n_items_min <= cfg.n_items_max:
        raise ValueError("scene config: need 1 <= n_items.min <= n_items.max")
    if any(v <= 0 for v in sizes.values()) or cfg.item_height <= 0:
        raise ValueError("scene config: sizes and item_height must be positive")
    for i, a in enumerate(boxes):
        if a.size <= 0:
            raise ValueError(f"scene config: box {a.label} needs a positive size")
        for b in boxes[i + 1 :]:
            if abs(a.x - b.x) < (a.size + b.size) / 2 and abs(a.y - b.y) < (a.size + b.size) / 2:
                raise ValueError(f"scene config: boxes {a.label} and {b.label} overlap")
    e, ia = cfg.extent, items_area
    if not (e.x_min <= ia.x_min and ia.x_max <= e.x_max and e.y_min <= ia.y_min and ia.y_max <= e.y_max):
        raise ValueError("scene config: items_area must lie inside the table extent")
    return cfg


def default_config_path() -> Path:
    """Path of the installed ``scene.yaml`` (falls back to the source tree)."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("sorting_scene")) / "config" / "scene.yaml"
    except ImportError, LookupError:
        return Path(__file__).resolve().parent.parent / "config" / "scene.yaml"
