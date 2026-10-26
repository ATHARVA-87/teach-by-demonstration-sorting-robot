"""Helpers shared by every node that sends or receives camera images.

Why not cv_bridge? cv_bridge is the usual ROS <-> OpenCV converter, but it is a
compiled library linked against the *system* OpenCV. Our nodes run in a conda
env with pip's OpenCV, and mixing the two can crash on mismatched native libs.
A sensor_msgs/Image is just raw bytes plus a shape, so numpy does the job.
"""

import numpy as np
from builtin_interfaces.msg import Time
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# --- QoS for image topics ---------------------------------------------------
# depth=1 (KEEP_LAST): only the newest frame matters for live video; older
# frames are replaced instead of queuing up and adding latency.
#
# RELIABLE, not BEST_EFFORT: a 640x480 frame is ~900 KB, which DDS splits into
# hundreds of UDP fragments. With BEST_EFFORT, losing any one fragment drops
# the whole frame. Measured on this machine (Fast DDS): BEST_EFFORT delivered
# 0.2 Hz, RELIABLE delivered the full 20 Hz. With depth=1, RELIABLE never
# builds a backlog, so it adds no latency here.
IMAGE_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)


def cv2_to_imgmsg(frame: np.ndarray, stamp: Time, frame_id: str = "camera") -> Image:
    """Pack an OpenCV BGR image (H x W x 3, uint8) into a sensor_msgs/Image.

    `stamp` should be the time the frame was captured: downstream nodes copy
    it forward so we can measure end-to-end latency.
    """
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = "bgr8"  # OpenCV's native channel order
    msg.is_bigendian = 0
    msg.step = msg.width * 3  # bytes per image row
    # ascontiguousarray guarantees the memory layout matches `step`.
    msg.data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
    return msg


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    """Unpack a bgr8/rgb8 sensor_msgs/Image into an OpenCV BGR array (a copy)."""
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"Unsupported image encoding '{msg.encoding}' (expected bgr8 or rgb8)")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    # Rows may be padded (step > width*3), so slice each row to its real width.
    frame = buf.reshape(msg.height, msg.step)[:, : msg.width * 3]
    frame = frame.reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        frame = frame[:, :, ::-1]  # RGB -> BGR
    return frame.copy()  # writable copy, safe to draw on
