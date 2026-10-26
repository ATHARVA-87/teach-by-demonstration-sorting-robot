"""Synthetic MediaPipe-like hand landmarks for tests and headless demos.

No webcam is needed: a scripted "hand" moves to an item, pinches, drags it
to a box and lets go. The geometry is simplified but keeps what the pinch
logic uses (wrist, middle MCP, thumb tip, index tip) realistic, including
the hand-size scaling when the hand moves toward or away from the camera.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

# Offsets of the 21 landmarks in "hand units" (hand size = wrist..middle MCP = 1),
# for an upright hand with the wrist below the knuckles (image y grows down).
_BASE = np.array(
    [
        [0.00, 1.00],  # 0 wrist
        [-0.35, 0.85],
        [-0.55, 0.65],
        [-0.65, 0.45],
        [-0.70, 0.30],  # 1-4 thumb (tip replaced)
        [-0.25, 0.05],
        [-0.28, -0.35],
        [-0.30, -0.60],
        [-0.30, -0.80],  # 5-8 index (tip replaced)
        [0.00, 0.00],
        [0.00, -0.45],
        [0.00, -0.72],
        [0.00, -0.95],  # 9-12 middle
        [0.22, 0.05],
        [0.24, -0.38],
        [0.25, -0.62],
        [0.25, -0.82],  # 13-16 ring
        [0.42, 0.15],
        [0.46, -0.20],
        [0.48, -0.40],
        [0.48, -0.55],  # 17-20 pinky
    ]
)


def make_hand(pinch_point: tuple[float, float], hand_size_px: float, ratio: float) -> np.ndarray:
    """21x2 pixel landmarks whose thumb/index tips meet at ``pinch_point``.

    Args:
        pinch_point: (u, v) pixel position between thumb and index tips.
        hand_size_px: wrist-to-middle-MCP length in pixels.
        ratio: desired pinch ratio (tip distance / hand size).
    """
    s = hand_size_px
    p = np.asarray(pinch_point, dtype=float)
    # Place the hand so its thumb/index meeting point is at p: the meeting
    # point sits up and to the left of the middle MCP.
    meet_offset = np.array([-0.55, -0.15])
    origin = p - meet_offset * s
    lm = origin + _BASE * s
    half = ratio * s / 2.0
    lm[4] = p + np.array([0.0, half])  # thumb tip below the meeting point
    lm[8] = p - np.array([0.0, half])  # index tip above it
    return lm


@dataclass(frozen=True)
class Frame:
    """One synthetic frame: time and landmarks (None = hand not visible)."""

    t: float
    landmarks: np.ndarray | None


def drag_sequence(
    start: tuple[float, float],
    end: tuple[float, float],
    fps: float = 30.0,
    hand_size_px: float = 90.0,
    t0: float = 0.0,
    open_ratio: float = 0.8,
    closed_ratio: float = 0.12,
    noise_px: float = 0.0,
    seed: int = 0,
    drop_out: tuple[int, int] | None = None,
) -> Iterator[Frame]:
    """Hover over ``start``, pinch, drag to ``end``, release, move away.

    Args:
        drop_out: optional (first, last) frame indices during the drag where
            the hand "disappears" (simulates tracking loss).
    """
    rng = np.random.default_rng(seed)
    phases: list[tuple[np.ndarray, float]] = []
    a, b = np.asarray(start, float), np.asarray(end, float)
    phases += [(a, open_ratio)] * 6  # hover
    phases += [(a, r) for r in np.linspace(open_ratio, closed_ratio, 5)]  # close
    phases += [(a, closed_ratio)] * 3  # hold
    phases += [(a + (b - a) * k, closed_ratio) for k in np.linspace(0, 1, 20)]  # drag
    phases += [(b, closed_ratio)] * 3
    phases += [(b, r) for r in np.linspace(closed_ratio, open_ratio, 5)]  # open
    phases += [(b + np.array([0.0, 60.0]), open_ratio)] * 6  # move away
    for i, (p, r) in enumerate(phases):
        t = t0 + i / fps
        if drop_out and drop_out[0] <= i <= drop_out[1]:
            yield Frame(t, None)
            continue
        lm = make_hand((float(p[0]), float(p[1])), hand_size_px, float(r))
        if noise_px:
            lm = lm + rng.normal(0.0, noise_px, lm.shape)
        yield Frame(t, lm)
