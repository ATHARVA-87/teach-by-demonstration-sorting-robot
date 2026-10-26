"""AR overlay renderer: draws the virtual cell over the webcam frame.

Follows docs/design_plan.md section 5 (decision D6):
- Item paint only on item glyphs; UI strokes are Signal white with a 1 px
  dark halo; no status hues inside the image.
- Boxes: Anthracite trays identified by letter + fill pattern (by slot index).
- Hover: corner brackets. Grab: thicker brackets + lifted chip with shadow.
- Query items: dashed brackets + "?" tag. Drop target: 3 px outline + inverted letter.
- Cursor: ring at the pinch point, filled when pinched, arc = pinch strength.
- Symbols are drawn as primitives (never font glyphs).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from ar_teach.palette import ANTHRACITE, BGR, BLACK_GREY, INK, SIGNAL_WHITE
from ar_teach.pinch import PinchState
from ar_teach.session import TeachSession, VirtualItem
from ar_teach.text_render import TextRenderer
from sorting_scene import Box, Rect, SceneConfig

AA = cv2.LINE_AA
PATTERNS = ("hatch", "dots", "plain", "crosshatch", "vertical")


def _poly(cfg: SceneConfig, pts_world: Sequence[tuple[float, float]]) -> np.ndarray:
    return np.array([cfg.ar_view.world_to_px(x, y) for x, y in pts_world], np.float32)


def _rect_poly(cfg: SceneConfig, r: Rect) -> np.ndarray:
    return _poly(cfg, [(r.x_min, r.y_min), (r.x_min, r.y_max), (r.x_max, r.y_max), (r.x_max, r.y_min)])


def _box_poly(cfg: SceneConfig, b: Box) -> np.ndarray:
    h = b.size / 2
    return _poly(cfg, [(b.x - h, b.y - h), (b.x - h, b.y + h), (b.x + h, b.y + h), (b.x + h, b.y - h)])


def _blend_fill(frame: np.ndarray, poly: np.ndarray, color: BGR, alpha: float) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly.round().astype(np.int32)], color, AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _halo_polyline(frame: np.ndarray, pts: np.ndarray, closed: bool, thickness: int = 1) -> None:
    """Signal-white polyline with a 1 px dark halo (design plan rev. 11)."""
    p = [pts.round().astype(np.int32)]
    cv2.polylines(frame, p, closed, INK, thickness + 2, AA)
    cv2.polylines(frame, p, closed, SIGNAL_WHITE, thickness, AA)


def _halo_line(frame: np.ndarray, a: tuple[int, int], b: tuple[int, int], thickness: int) -> None:
    cv2.line(frame, a, b, INK, thickness + 2, AA)
    cv2.line(frame, a, b, SIGNAL_WHITE, thickness, AA)


def pattern_for_slot(slot: int) -> str:
    """Box fill pattern by slot index, repeating after the defined set."""
    return PATTERNS[slot % len(PATTERNS)]


class OverlayRenderer:
    """Draws one :class:`TeachSession` frame over a webcam image."""

    def __init__(self, scene_cfg: SceneConfig, font_path: str | None) -> None:
        self.cfg = scene_cfg
        self.text = TextRenderer(font_path)
        self._paint = {name: (rgb[2], rgb[1], rgb[0]) for name, rgb in scene_cfg.colors.items()}

    # --- public -----------------------------------------------------------------
    def draw(
        self, frame: np.ndarray, session: TeachSession, pinch: PinchState, banner: str | None = None
    ) -> None:
        """Draw the whole overlay onto ``frame`` (BGR, in place)."""
        self._draw_table(frame)
        target = session.drop_target()
        for b in self.cfg.boxes:
            self._draw_box(frame, b, highlighted=(b.label == target))
        self._draw_boxed_items(frame, session)
        for vi in session.table_items():
            held = vi.item.id == session.held
            self._draw_item(frame, vi, lifted=held)
            if vi.is_query:
                self._draw_brackets(frame, vi, dashed=True, thickness=2)
                self._draw_query_tag(frame, vi)
            if vi.item.id == session.hover:
                self._draw_brackets(frame, vi, dashed=False, thickness=3 if held else 1)
        self._draw_cursor(frame, pinch, holding=session.held is not None)
        if banner:
            self._draw_banner(frame, banner)

    # --- elements -----------------------------------------------------------------
    def _draw_table(self, frame: np.ndarray) -> None:
        poly = _rect_poly(self.cfg, self.cfg.extent)
        _blend_fill(frame, poly, BLACK_GREY, 0.35)
        _halo_polyline(frame, poly, closed=True)

    def _draw_box(self, frame: np.ndarray, b: Box, highlighted: bool) -> None:
        poly = _box_poly(self.cfg, b)
        _blend_fill(frame, poly, ANTHRACITE, 0.85)
        self._draw_pattern(frame, poly, pattern_for_slot(b.slot))
        _halo_polyline(frame, poly, closed=True, thickness=3 if highlighted else 1)
        # Letter in the tray's top-left corner, so it never sits under an
        # item being dropped in the middle.
        size = max(18, int(self.cfg.ar_view.meters_to_px(b.size) * 0.30))
        x0, y0 = poly.min(axis=0)
        lu, lv = int(x0 + size * 0.7), int(y0 + size * 0.75)
        if highlighted:  # inverted letter: ink on a white tile
            half = int(size * 0.55)
            cv2.rectangle(frame, (lu - half - 1, lv - half - 1), (lu + half + 1, lv + half + 1), INK, -1, AA)
            cv2.rectangle(frame, (lu - half, lv - half), (lu + half, lv + half), SIGNAL_WHITE, -1, AA)
            self.text.draw(frame, b.label, (lu, lv), size, INK, halo=SIGNAL_WHITE)
        else:
            self.text.draw(frame, b.label, (lu, lv), size, SIGNAL_WHITE)

    def _draw_pattern(self, frame: np.ndarray, poly: np.ndarray, pattern: str) -> None:
        x0, y0 = poly.min(axis=0).astype(int)
        x1, y1 = poly.max(axis=0).astype(int)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly.round().astype(np.int32)], 255)
        layer = frame.copy()
        step = 9
        if pattern in ("hatch", "crosshatch"):
            for k in range(-(y1 - y0), x1 - x0, step):
                cv2.line(layer, (x0 + k, y1), (x0 + k + (y1 - y0), y0), SIGNAL_WHITE, 1, AA)
        if pattern == "crosshatch":
            for k in range(0, (x1 - x0) + (y1 - y0), step):
                cv2.line(layer, (x0 + k, y0), (x0 + k - (y1 - y0), y1), SIGNAL_WHITE, 1, AA)
        if pattern == "dots":
            for yy in range(y0 + step // 2, y1, step):
                for xx in range(x0 + step // 2, x1, step):
                    cv2.circle(layer, (xx, yy), 1, SIGNAL_WHITE, -1, AA)
        if pattern == "vertical":
            for xx in range(x0 + step // 2, x1, step):
                cv2.line(layer, (xx, y0), (xx, y1), SIGNAL_WHITE, 1, AA)
        blended = cv2.addWeighted(layer, 0.30, frame, 0.70, 0)
        frame[mask > 0] = blended[mask > 0]

    def _item_geometry(self, vi: VirtualItem, du: float = 0.0, dv: float = 0.0) -> tuple[float, float, float]:
        u, v = self.cfg.ar_view.world_to_px(vi.item.x, vi.item.y)
        r = self.cfg.ar_view.meters_to_px(self.cfg.sizes[vi.item.size] / 2.0)
        return u + du, v + dv, r

    def _glyph_points(self, shape: str, u: float, v: float, r: float, yaw: float) -> np.ndarray:
        if shape == "cube":
            corners = [(-r, -r), (r, -r), (r, r), (-r, r)]
        elif shape == "prism":  # equilateral triangle inscribed in radius r * 1.15
            rr = r * 1.15
            corners = [
                (rr * math.cos(a), rr * math.sin(a)) for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)
            ]
        else:
            raise ValueError(shape)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([(u + c * x - s * y, v + s * x + c * y) for x, y in corners], np.float32)

    def _draw_glyph(
        self, frame: np.ndarray, shape: str, u: float, v: float, r: float, yaw: float, fill: BGR
    ) -> None:
        if shape == "cylinder":
            cv2.circle(frame, (round(u), round(v)), round(r), fill, -1, AA)
            cv2.circle(frame, (round(u), round(v)), round(r), INK, 2, AA)
            return
        pts = self._glyph_points(shape, u, v, r, yaw).round().astype(np.int32)
        cv2.fillPoly(frame, [pts], fill, AA)
        cv2.polylines(frame, [pts], True, INK, 2, AA)

    def _draw_item(self, frame: np.ndarray, vi: VirtualItem, lifted: bool) -> None:
        u, v, r = self._item_geometry(vi)
        paint = self._paint[vi.item.color]
        if lifted:  # small offset shadow, chip raised toward the viewer
            overlay = frame.copy()
            self._draw_glyph(overlay, vi.item.shape, u + 5, v + 6, r, vi.item.yaw, INK)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            u, v = u - 3, v - 4
        self._draw_glyph(frame, vi.item.shape, u, v, r, vi.item.yaw, paint)

    def _draw_boxed_items(self, frame: np.ndarray, session: TeachSession) -> None:
        """Small chips inside each box, in a grid, so the worker sees what's there."""
        for b in self.cfg.boxes:
            inside = [vi for vi in session.items.values() if vi.status == "boxed" and vi.box == b.label]
            if not inside:
                continue
            bu, bv = self.cfg.ar_view.world_to_px(b.x, b.y)
            half = self.cfg.ar_view.meters_to_px(b.size) / 2
            cell = half * 2 / 4
            for k, vi in enumerate(inside[:8]):
                col, row = k % 4, k // 4
                u = bu - half + cell * (col + 0.5)
                v = bv + half - cell * (row + 0.5)
                self._draw_glyph(frame, vi.item.shape, u, v, cell * 0.34, 0.0, self._paint[vi.item.color])

    def _draw_brackets(self, frame: np.ndarray, vi: VirtualItem, dashed: bool, thickness: int) -> None:
        """Inspection-style corner brackets around an item (dashed for questions)."""
        u, v, r = self._item_geometry(vi)
        if vi.status == "held":
            u, v = u - 3, v - 4
        h = r * 1.45 + 6
        arm = max(8.0, h * 0.55)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cx, cy = u + sx * h, v + sy * h
            for ex, ey in ((cx - sx * arm, cy), (cx, cy - sy * arm)):
                if not dashed:
                    _halo_line(frame, (round(cx), round(cy)), (round(ex), round(ey)), thickness)
                    continue
                # Two dashes per arm: [0, 0.4] and [0.6, 1.0] of its length.
                for t0, t1 in ((0.0, 0.4), (0.6, 1.0)):
                    p = (round(cx + (ex - cx) * t0), round(cy + (ey - cy) * t0))
                    q = (round(cx + (ex - cx) * t1), round(cy + (ey - cy) * t1))
                    _halo_line(frame, p, q, thickness)

    def _draw_query_tag(self, frame: np.ndarray, vi: VirtualItem) -> None:
        """'?' tag on a dark disc, outside the top-right bracket."""
        u, v, r = self._item_geometry(vi)
        h = r * 1.45 + 6
        tag_u, tag_v = round(u + h + 16), round(v - h - 14)
        cv2.circle(frame, (tag_u, tag_v), 12, INK, -1, AA)
        cv2.circle(frame, (tag_u, tag_v), 11, BLACK_GREY, -1, AA)
        cv2.circle(frame, (tag_u, tag_v), 11, SIGNAL_WHITE, 1, AA)
        self.text.draw(frame, "?", (tag_u, tag_v), 17, SIGNAL_WHITE)

    def _draw_cursor(self, frame: np.ndarray, pinch: PinchState, holding: bool = False) -> None:
        if not pinch.tracked or pinch.cursor is None:
            return
        c = (round(pinch.cursor[0]), round(pinch.cursor[1]))
        if holding:  # hollow ring only, so a small held item stays visible
            cv2.circle(frame, c, 5, INK, 3, AA)
            cv2.circle(frame, c, 5, SIGNAL_WHITE, 1, AA)
        elif pinch.pinched:
            cv2.circle(frame, c, 8, INK, -1, AA)
            cv2.circle(frame, c, 7, SIGNAL_WHITE, -1, AA)
        else:
            cv2.circle(frame, c, 9, INK, 4, AA)
            cv2.circle(frame, c, 9, SIGNAL_WHITE, 2, AA)
        sweep = int(360 * pinch.strength)
        if sweep > 0 and not holding:  # pinch-strength arc
            cv2.ellipse(frame, c, (15, 15), -90, 0, sweep, INK, 4, AA)
            cv2.ellipse(frame, c, (15, 15), -90, 0, sweep, SIGNAL_WHITE, 2, AA)

    def _draw_banner(self, frame: np.ndarray, text: str) -> None:
        w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 34), BLACK_GREY, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        self.text.draw(frame, text, (w // 2, 17), 17, SIGNAL_WHITE)
