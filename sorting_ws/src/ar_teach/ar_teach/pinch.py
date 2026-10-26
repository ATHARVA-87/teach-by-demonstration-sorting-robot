"""Pinch detection with hysteresis, smoothing and debouncing.

Input: 21 MediaPipe hand landmarks in **pixels**. Output: whether the hand
is pinching, plus a smoothed cursor at the pinch point.

    ratio = |thumb_tip - index_tip| / |wrist - middle_mcp|

Dividing by the wrist-to-middle-knuckle length (a "hand size" that doesn't
change when fingers move) makes the ratio independent of how far the hand
is from the camera.

Hysteresis: the hand becomes *closed* when ratio < grab_ratio and *open*
again only when ratio > release_ratio (> grab_ratio). Between the two
thresholds the state is kept, so noise around one threshold can't make the
grab flicker. A debouncer then requires the new state for a few frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sorting_common.filters import Debouncer, Ema

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
OPEN, CLOSED = "OPEN", "CLOSED"


@dataclass(frozen=True)
class PinchConfig:
    """Thresholds and filter settings (from ar_teach YAML)."""

    grab_ratio: float = 0.30  # close below this
    release_ratio: float = 0.45  # open above this (must be > grab_ratio)
    ratio_alpha: float = 0.6  # EMA on the ratio (1 = no smoothing)
    cursor_alpha: float = 0.5  # EMA on the cursor position
    grab_frames: int = 2  # frames of CLOSED needed to grab
    release_frames: int = 2  # frames of OPEN needed to release

    def __post_init__(self) -> None:
        if not 0.0 < self.grab_ratio < self.release_ratio:
            raise ValueError("need 0 < grab_ratio < release_ratio (hysteresis band)")
        if self.grab_frames < 1 or self.release_frames < 1:
            raise ValueError("debounce frame counts must be >= 1")


@dataclass(frozen=True)
class PinchState:
    """Result of one update."""

    tracked: bool  # a hand was seen this frame
    pinched: bool  # debounced pinch state
    ratio: float | None  # smoothed pinch ratio
    cursor: tuple[float, float] | None  # smoothed pinch point (px)
    strength: float  # 0 = open, 1 = fully pinched (for the cursor arc)


def pinch_ratio(landmarks: np.ndarray) -> float:
    """Thumb-index tip distance divided by hand size (wrist to middle MCP)."""
    hand_size = float(np.linalg.norm(landmarks[WRIST] - landmarks[MIDDLE_MCP]))
    if hand_size < 1e-6:
        raise ValueError("degenerate hand: wrist and middle MCP coincide")
    return float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP])) / hand_size


def pinch_point(landmarks: np.ndarray) -> tuple[float, float]:
    """Midpoint of thumb and index tips: where the 'fingers' meet."""
    p = (landmarks[THUMB_TIP] + landmarks[INDEX_TIP]) / 2.0
    return float(p[0]), float(p[1])


class PinchDetector:
    """Stateful pinch detector; call :meth:`update` once per frame."""

    def __init__(self, cfg: PinchConfig | None = None) -> None:
        self.cfg = cfg or PinchConfig()
        self._ratio = Ema(self.cfg.ratio_alpha)
        self._cu = Ema(self.cfg.cursor_alpha)
        self._cv = Ema(self.cfg.cursor_alpha)
        self._hyst = OPEN  # raw state after hysteresis, before debouncing
        self._debounce = Debouncer(OPEN, self._frames_for)
        self._last = PinchState(False, False, None, None, 0.0)

    def _frames_for(self, state: str) -> int:
        return self.cfg.grab_frames if state == CLOSED else self.cfg.release_frames

    def reset(self) -> None:
        """Forget all history (e.g. after tracking has been lost for a while)."""
        self._ratio.reset()
        self._cu.reset()
        self._cv.reset()
        self._hyst = OPEN
        self._debounce.force(OPEN)
        self._last = PinchState(False, False, None, None, 0.0)

    def update(self, landmarks: np.ndarray | None) -> PinchState:
        """Process one frame. ``None`` means no hand was detected.

        A missing hand does not change the pinch state by itself (a one-frame
        tracking dropout mid-drag must not drop the item); the caller decides
        when a dropout lasts too long and calls :meth:`reset`.
        """
        if landmarks is None:
            last = self._last
            self._last = PinchState(False, last.pinched, last.ratio, last.cursor, last.strength)
            return self._last

        lm = np.asarray(landmarks, dtype=float)
        ratio = self._ratio.update(pinch_ratio(lm))
        if self._hyst == OPEN and ratio < self.cfg.grab_ratio:
            self._hyst = CLOSED
        elif self._hyst == CLOSED and ratio > self.cfg.release_ratio:
            self._hyst = OPEN
        pinched = self._debounce.update(self._hyst) == CLOSED

        u, v = pinch_point(lm)
        cursor = (self._cu.update(u), self._cv.update(v))
        floor = self.cfg.grab_ratio * 0.5
        strength = float(np.clip((self.cfg.release_ratio - ratio) / (self.cfg.release_ratio - floor), 0, 1))
        self._last = PinchState(True, pinched, ratio, cursor, strength)
        return self._last
