"""Webcam capture in a background thread (ported from the gesture project).

``cap.read()`` blocks until the next frame, so a dedicated thread reads at
the camera's own rate and keeps only the newest frame (no queue, no extra
latency). Lessons kept from the gesture project (WALKTHROUGH troubleshooting):
- MJPG format: USB webcams only reach 30 fps compressed.
- Never set CAP_PROP_BUFFERSIZE=1: with one V4L2 buffer the driver drops
  every other frame (measured 30 -> 15 fps).
- Lock ``exposure_dynamic_framerate=0`` via v4l2-ctl so dim light doesn't
  drop the frame rate; the original value is restored on close.
- Reconnect automatically if the camera disappears.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

MSG_NO_WEBCAM = "No webcam found. Plug one in or close other apps using it."

Logger = Callable[[str, str], None]  # (level, message)


@dataclass(frozen=True)
class WebcamConfig:
    """Capture settings (from ar_teach YAML)."""

    index: int = 0
    width: int = 640
    height: int = 480
    fps: float = 30.0
    mirror: bool = True  # mirror view: moving right moves right on screen
    lock_framerate: bool = True
    reopen_period_s: float = 2.0  # retry interval when the camera is missing
    max_failed_reads: int = 15  # consecutive failed reads that count as "camera lost"


class WebcamSource:
    """Latest-frame webcam reader. Thread-safe :meth:`latest`."""

    def __init__(self, cfg: WebcamConfig, log: Logger) -> None:
        self.cfg = cfg
        self._log = log
        self._reported_missing = False
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self.available = False
        self._original_dynamic_fps: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="webcam", daemon=True)
        self._thread.start()

    def latest(self) -> tuple[int, float, np.ndarray | None]:
        """(sequence number, capture time, frame) of the newest frame."""
        with self._lock:
            return self._seq, self._stamp, self._frame

    def close(self) -> None:
        """Stop the thread, release the camera and restore its settings."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._original_dynamic_fps is not None:
            self._v4l2("-c", f"exposure_dynamic_framerate={self._original_dynamic_fps}")

    # --- internals ------------------------------------------------------------
    def _v4l2(self, *args: str) -> str | None:
        if shutil.which("v4l2-ctl") is None:
            return None
        cmd = ["v4l2-ctl", "-d", f"/dev/video{self.cfg.index}", *args]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
        return result.stdout if result.returncode == 0 else None

    def _lock_framerate(self) -> None:
        current = self._v4l2("--get-ctrl", "exposure_dynamic_framerate")
        if current is None:
            self._log(
                "warning",
                "Could not lock the webcam frame rate (v4l2-ctl missing or control "
                "unsupported). In dim light the frame rate may drop.",
            )
            return
        if self._original_dynamic_fps is None:
            self._original_dynamic_fps = current.split(":")[-1].strip()
        self._v4l2("-c", "exposure_dynamic_framerate=0")

    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.cfg.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            if self._reported_missing:  # report once, then stay quiet until it changes
                return None
            self._reported_missing = True
            self._log(
                "error",
                f"{MSG_NO_WEBCAM} (camera index {self.cfg.index}, "
                f"retrying every {self.cfg.reopen_period_s:.0f} s)",
            )
            return None
        self._reported_missing = False
        if self.cfg.lock_framerate:
            self._lock_framerate()
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._log("info", f"Webcam {self.cfg.index} opened at {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
        return cap

    def _loop(self) -> None:
        """Capture loop. One iteration failing (driver error, etc.) never kills the thread."""
        cap: cv2.VideoCapture | None = None
        failed = 0
        while not self._stop.is_set():
            try:
                cap, failed = self._step(cap, failed)
            except Exception as exc:  # keep the camera thread alive; report and retry
                self._log("error", f"Webcam error ({type(exc).__name__}: {exc}). Reconnecting.")
                if cap is not None:
                    cap.release()
                cap, failed = None, 0
                self.available = False
                self._stop.wait(self.cfg.reopen_period_s)
        if cap is not None:
            cap.release()

    def _step(self, cap: cv2.VideoCapture | None, failed: int) -> tuple[cv2.VideoCapture | None, int]:
        """One capture iteration; returns the (possibly new) capture and failure count."""
        if cap is None:
            cap = self._open()
            self.available = cap is not None
            if cap is None:
                self._stop.wait(self.cfg.reopen_period_s)
                return None, 0
            failed = 0
        ok, frame = cap.read()
        stamp = time.time()
        if not ok or frame is None:
            failed += 1
            if failed >= self.cfg.max_failed_reads:
                self._log("error", "Webcam stopped sending frames. Check the USB cable; reconnecting.")
                cap.release()
                self.available = False
                return None, 0
            time.sleep(0.01)
            return cap, failed
        if frame.shape[1] != self.cfg.width or frame.shape[0] != self.cfg.height:
            frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        with self._lock:
            self._frame, self._stamp = frame, stamp
            self._seq += 1
        return cap, 0
