"""Lighting check for the webcam frame: is it good enough to track a hand?"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MSG_TOO_DARK = "Too dark to track your hand. Turn on a light or face a window."
MSG_TOO_BRIGHT = "Too much glare to track your hand. Move away from direct light."


@dataclass(frozen=True)
class LightingConfig:
    """Thresholds on mean luma (0-255) and the share of clipped pixels."""

    dark_mean: float = 45.0
    bright_clipped_fraction: float = 0.35


def lighting_message(frame_bgr: np.ndarray, cfg: LightingConfig | None = None) -> str | None:
    """Return a worker-facing message if lighting is poor, else None."""
    cfg = cfg or LightingConfig()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < cfg.dark_mean:
        return MSG_TOO_DARK
    if float((gray >= 250).mean()) > cfg.bright_clipped_fraction:
        return MSG_TOO_BRIGHT
    return None
