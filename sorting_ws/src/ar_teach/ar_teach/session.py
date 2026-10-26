"""Teaching session: hover, grab, drag and drop of virtual items (no ROS).

Feeds on :class:`~ar_teach.pinch.PinchState` once per frame and produces
events. Items live in world coordinates (``sorting_scene``); the AR view is
just a projection through :class:`~sorting_scene.ArView`, so a drop is
decided in the same metric space the robot uses.

Edge cases (from the spec):
- **Overlapping items:** the item whose center is closest to the cursor wins;
  ties go to the item drawn on top (most recently moved).
- **Dropping between boxes / off the table:** the item returns to where it
  was picked up, with a message.
- **Tracking lost mid-drag** (no hand for ``lost_timeout_s``): the item
  returns to its start position, with a message. Shorter dropouts are bridged.
- **Pinching in empty space, or sliding onto an item while already pinched:**
  no grab. A grab needs the pinch to *start* over an item.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ar_teach.pinch import PinchState
from sorting_scene import Item, SceneConfig

# Messages (copy from docs/design_plan.md, section 5).
MSG_HAND_LOST = "Hand lost. Item returned to its place."
MSG_BETWEEN_BOXES = "Dropped between boxes. Item returned to the table."
MSG_OFF_TABLE = "Dropped off the table. Item returned to its place."


@dataclass(frozen=True)
class SessionConfig:
    """Interaction tuning (from ar_teach YAML)."""

    hover_margin_px: float = 8.0  # extra pick radius around an item
    lost_timeout_s: float = 0.4  # dropout longer than this cancels a drag


@dataclass
class VirtualItem:
    """An item in the AR scene and where it currently is."""

    item: Item  # attributes + current world position
    home: tuple[float, float]  # world XY where the current/last grab started
    status: str = "table"  # "table" | "held" | "boxed"
    box: str | None = None  # box label once boxed
    is_query: bool = False  # placed by the rule learner as a question
    z: int = 0  # drawing order (higher = on top)


@dataclass(frozen=True)
class DemoEvent:
    """A completed demonstration: this item went into this box."""

    item: Item  # attributes; position = where it was picked up
    box: str
    t: float  # time of the drop
    duration_s: float  # grab -> drop
    is_query: bool


@dataclass(frozen=True)
class MessageEvent:
    """Something to tell the worker on screen."""

    level: str  # "info" | "warning"
    text: str
    t: float


Event = DemoEvent | MessageEvent


@dataclass
class TeachSession:
    """State machine for one teaching session."""

    scene_cfg: SceneConfig
    cfg: SessionConfig = field(default_factory=SessionConfig)
    items: dict[str, VirtualItem] = field(default_factory=dict)
    held: str | None = None
    hover: str | None = None
    _offset: tuple[float, float] = (0.0, 0.0)
    _grab_t: float = 0.0
    _was_pinched: bool = False
    _armed: bool = True  # False after a reset until the hand is seen open once
    _last_seen: float | None = None
    _z: int = 0

    # --- scene management ---------------------------------------------------
    def add_item(self, item: Item, is_query: bool = False) -> None:
        """Put an item on the table (e.g. a scene item or an active-learning query)."""
        self._z += 1
        self.items[item.id] = VirtualItem(
            item=replace(item), home=(item.x, item.y), is_query=is_query, z=self._z
        )

    def remove_item(self, item_id: str) -> None:
        """Remove an item entirely (e.g. when a demonstration is deleted)."""
        if self.held == item_id:
            self.held = None
        self.items.pop(item_id, None)

    def table_items(self) -> list[VirtualItem]:
        """Items on the table or in the hand, in drawing order."""
        return sorted((v for v in self.items.values() if v.status != "boxed"), key=lambda v: v.z)

    # --- geometry -----------------------------------------------------------
    def _item_px(self, v: VirtualItem) -> tuple[float, float]:
        return self.scene_cfg.ar_view.world_to_px(v.item.x, v.item.y)

    def _radius_px(self, v: VirtualItem) -> float:
        return self.scene_cfg.ar_view.meters_to_px(self.scene_cfg.sizes[v.item.size] / 2.0)

    def item_under(self, u: float, v: float) -> str | None:
        """Item to pick at pixel (u, v): nearest center within reach, top-most on ties."""
        best: tuple[float, int, str] | None = None
        for vi in self.table_items():
            cu, cv = self._item_px(vi)
            d = math.hypot(u - cu, v - cv)
            if d <= self._radius_px(vi) + self.cfg.hover_margin_px:
                key = (d, -vi.z, vi.item.id)
                if best is None or key < best:
                    best = key
        return best[2] if best else None

    # --- per-frame update ---------------------------------------------------
    def update(self, t: float, pinch: PinchState) -> list[Event]:
        """Advance one frame and return the events it produced."""
        events: list[Event] = []
        if not pinch.tracked or pinch.cursor is None:
            self.hover = None
            lost_for = t - self._last_seen if self._last_seen is not None else math.inf
            if self.held is not None and lost_for > self.cfg.lost_timeout_s:
                self._return_home(self.held)
                events.append(MessageEvent("warning", MSG_HAND_LOST, t))
                self._was_pinched = False
                self._armed = False  # a hand that comes back already pinched must not grab
            return events

        self._last_seen = t
        u, v = pinch.cursor
        if not pinch.pinched:
            self._armed = True
        rising = pinch.pinched and not self._was_pinched and self._armed
        falling = self._was_pinched and not pinch.pinched
        self._was_pinched = pinch.pinched

        if self.held is None:
            self.hover = self.item_under(u, v)
            if rising and self.hover is not None:
                self._grab(self.hover, u, v, t)
            return events

        held = self.items[self.held]
        x, y = self.scene_cfg.ar_view.px_to_world(u + self._offset[0], v + self._offset[1])
        held.item.x, held.item.y = x, y
        self.hover = self.held
        if falling:
            events.extend(self._drop(t))
        return events

    def drop_target(self) -> str | None:
        """Label of the box under the held item, if any (for highlighting)."""
        if self.held is None:
            return None
        it = self.items[self.held].item
        box = self.scene_cfg.box_at(it.x, it.y)
        return box.label if box else None

    # --- transitions --------------------------------------------------------
    def _grab(self, item_id: str, u: float, v: float, t: float) -> None:
        vi = self.items[item_id]
        cu, cv = self._item_px(vi)
        self._offset = (cu - u, cv - v)  # keep the item where it was grabbed
        vi.home = (vi.item.x, vi.item.y)
        vi.status = "held"
        self._z += 1
        vi.z = self._z
        self.held = item_id
        self._grab_t = t

    def _return_home(self, item_id: str) -> None:
        vi = self.items[item_id]
        vi.item.x, vi.item.y = vi.home
        vi.status = "table"
        self.held = None

    def _drop(self, t: float) -> list[Event]:
        if self.held is None:
            return []
        vi = self.items[self.held]
        box = self.scene_cfg.box_at(vi.item.x, vi.item.y)
        if box is not None:
            demo_item = replace(vi.item, x=vi.home[0], y=vi.home[1])
            vi.status, vi.box = "boxed", box.label
            vi.item.x, vi.item.y = box.x, box.y
            self.held = None
            return [DemoEvent(demo_item, box.label, t, t - self._grab_t, vi.is_query)]
        radius = self.scene_cfg.sizes[vi.item.size] / 2.0
        if self.scene_cfg.items_area.contains(vi.item.x, vi.item.y, margin=radius):
            vi.status = "table"  # moved on the table: not a demonstration
            self.held = None
            return []
        near_boxes = any(
            abs(vi.item.x - b.x) <= b.size and abs(vi.item.y - b.y) <= b.size for b in self.scene_cfg.boxes
        )
        self._return_home(self.held)
        return [MessageEvent("info", MSG_BETWEEN_BOXES if near_boxes else MSG_OFF_TABLE, t)]

    def disarm(self) -> None:
        """Require an open hand before the next grab (call after resetting the pinch detector)."""
        self._armed = False
