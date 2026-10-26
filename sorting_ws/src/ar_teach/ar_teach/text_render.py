"""Translucent text for overlays drawn on video frames (Pillow + numpy).

Carried over from the gesture project's guide overlay; only the text part is
kept. Each (text, size) is rendered once with Pillow into an alpha mask plus
a blurred "halo" mask, cached, and alpha-blended onto the frame:
``pixel = pixel * (1 - a) + color * a`` (halo dark first, then the text).
The halo keeps text readable over bright and dark video alike.
"""

from __future__ import annotations

import functools
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HALO_PAD = 4


class TextRenderer:
    """Cached, alpha-blended text drawing in one font."""

    def __init__(self, font_path: str | None) -> None:
        self.font_path = font_path

    @functools.lru_cache(maxsize=512)  # noqa: B019 (one renderer per node; bounded cache)
    def masks(self, text: str, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Glyph mask and halo mask (float32, 0..1, same shape)."""
        if self.font_path and os.path.isfile(self.font_path):
            font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(self.font_path, size)
        else:
            font = ImageFont.load_default(size)  # fallback if the font is missing
        left, top, right, bottom = font.getbbox(text)
        pad = HALO_PAD
        img = Image.new("L", (int(right - left) + 2 * pad, int(bottom - top) + 2 * pad), 0)
        ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=255)
        mask = np.asarray(img, dtype=np.float32) / 255.0
        halo = cv2.GaussianBlur(cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2), (0, 0), 1.5)
        return mask, np.clip(halo * 1.4, 0.0, 1.0)

    def draw(
        self,
        frame: np.ndarray,
        text: str,
        xy: tuple[int, int],
        size: int,
        color: tuple[int, int, int],
        opacity: float = 1.0,
        anchor: str = "center",
        halo: tuple[int, int, int] = (36, 33, 30),
    ) -> None:
        """Alpha-blend ``text`` onto ``frame`` in place.

        Args:
            xy: anchor point in pixels.
            anchor: "center" (xy is the text center) or "left" (xy is the
                left edge, vertically centered).
            halo: dark outline color drawn first so text reads on any background.
        """
        mask, halo_mask = self.masks(text, size)
        th, tw = mask.shape
        x0 = xy[0] - tw // 2 if anchor == "center" else xy[0] - HALO_PAD
        y0 = xy[1] - th // 2
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + tw, fw), min(y0 + th, fh)
        if x1 >= x2 or y1 >= y2:
            return
        sl = (slice(y1 - y0, y2 - y0), slice(x1 - x0, x2 - x0))
        roi: np.ndarray = frame[y1:y2, x1:x2].astype(np.float32)
        a_halo = halo_mask[sl][..., None] * (0.6 * opacity)
        roi = roi * (1.0 - a_halo) + np.array(halo, np.float32) * a_halo
        a = mask[sl][..., None] * opacity
        roi = roi * (1.0 - a) + np.array(color, np.float32) * a
        frame[y1:y2, x1:x2] = roi.astype(np.uint8)
