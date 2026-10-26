"""Unit tests for sorting_common.image_utils (numpy <-> sensor_msgs/Image)."""

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image

from sorting_common.image_utils import cv2_to_imgmsg, imgmsg_to_cv2


def test_round_trip_bgr8() -> None:
    frame = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    msg = cv2_to_imgmsg(frame, Time(sec=5, nanosec=7), "cam")
    assert (msg.height, msg.width, msg.step, msg.encoding) == (48, 64, 192, "bgr8")
    assert msg.header.frame_id == "cam" and msg.header.stamp.sec == 5
    assert np.array_equal(imgmsg_to_cv2(msg), frame)


def test_padded_rows_are_cropped() -> None:
    h, w, step = 4, 5, 20  # 15 bytes of pixels + 5 bytes padding per row
    rows = np.zeros((h, step), np.uint8)
    rows[:, : w * 3] = 7
    msg = Image(height=h, width=w, step=step, encoding="bgr8", data=rows.tobytes())
    out = imgmsg_to_cv2(msg)
    assert out.shape == (h, w, 3) and np.all(out == 7)


def test_rgb8_is_converted_to_bgr() -> None:
    px = np.array([[[10, 20, 30]]], np.uint8)  # R, G, B
    msg = Image(height=1, width=1, step=3, encoding="rgb8", data=px.tobytes())
    assert imgmsg_to_cv2(msg)[0, 0].tolist() == [30, 20, 10]


def test_unsupported_encoding_raises() -> None:
    msg = Image(height=1, width=1, step=1, encoding="mono8", data=b"\x00")
    with pytest.raises(ValueError, match="mono8"):
        imgmsg_to_cv2(msg)
