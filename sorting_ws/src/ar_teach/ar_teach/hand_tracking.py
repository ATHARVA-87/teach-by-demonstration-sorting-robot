"""MediaPipe HandLandmarker wrapper (ported from the gesture project).

Uses the Tasks API in VIDEO mode: it tracks the hand from the previous frame
(faster and steadier than detecting from scratch) and needs strictly
increasing timestamps. Returns landmarks in **pixels** of the given frame.
"""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode


class ModelLoadError(RuntimeError):
    """The hand model file exists but MediaPipe can't load it (corrupt or wrong file)."""


MSG_NO_MODEL = (
    "Hand model not found at '{path}'. Run 'bash scripts/download_model.sh' from the "
    "repo root, or set the model_path parameter."
)


class HandTracker:
    """One-hand landmark tracker."""

    def __init__(self, model_path: str, min_detection: float = 0.5, min_tracking: float = 0.5) -> None:
        """Load the model.

        Raises:
            FileNotFoundError: the file doesn't exist (message says how to get it).
            ModelLoadError: the file exists but isn't a valid model.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(MSG_NO_MODEL.format(path=model_path))
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        try:
            self._landmarker = HandLandmarker.create_from_options(options)
        except (RuntimeError, ValueError) as exc:
            raise ModelLoadError(
                f"Hand model at '{model_path}' could not be loaded ({exc}). "
                "Re-run 'bash scripts/download_model.sh' or fix the model_path parameter."
            ) from exc
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, t_s: float) -> np.ndarray | None:
        """21x2 landmark pixel coordinates, or None if no hand is visible."""
        ts_ms = max(int(t_s * 1000), self._last_ts_ms + 1)  # strictly increasing
        self._last_ts_ms = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        if not result.hand_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([[lm.x * w, lm.y * h] for lm in result.hand_landmarks[0]], dtype=float)

    def close(self) -> None:
        """Free MediaPipe's native resources."""
        self._landmarker.close()
