"""The webcam thread survives driver exceptions and keeps delivering frames."""

import time

import numpy as np
import pytest

from ar_teach import webcam
from ar_teach.webcam import WebcamConfig, WebcamSource


class FlakyCapture:
    """Fake cv2.VideoCapture: the first read raises, later reads return frames."""

    calls = 0

    def __init__(self, *_args: object) -> None:
        pass

    def isOpened(self) -> bool:  # noqa: N802 (OpenCV API name)
        return True

    def set(self, *_args: object) -> bool:
        return True

    def get(self, *_args: object) -> float:
        return 30.0

    def read(self) -> tuple[bool, np.ndarray]:
        FlakyCapture.calls += 1
        if FlakyCapture.calls == 1:
            raise RuntimeError("simulated driver error")
        time.sleep(0.01)
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self) -> None:
        pass


def test_thread_survives_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webcam.cv2, "VideoCapture", FlakyCapture)
    logs: list[tuple[str, str]] = []
    src = WebcamSource(
        WebcamConfig(lock_framerate=False, reopen_period_s=0.05), lambda lv, m: logs.append((lv, m))
    )
    try:
        deadline = time.monotonic() + 3.0
        while src.latest()[0] < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert src.latest()[0] >= 3 and src.available
        assert any("simulated driver error" in m for lv, m in logs if lv == "error")
    finally:
        src.close()


class NoCamera(FlakyCapture):
    def isOpened(self) -> bool:  # noqa: N802
        return False


def test_missing_camera_reported_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webcam.cv2, "VideoCapture", NoCamera)
    logs: list[tuple[str, str]] = []
    src = WebcamSource(
        WebcamConfig(lock_framerate=False, reopen_period_s=0.02), lambda lv, m: logs.append((lv, m))
    )
    try:
        time.sleep(0.3)  # many retries
        assert not src.available
        assert sum("No webcam found" in m for _, m in logs) == 1
    finally:
        src.close()
