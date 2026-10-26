"""ar_teach node: AR teaching on the webcam view.

Pipeline per frame (timer at ``fps``):
    frame (webcam or synthetic) -> hand landmarks (MediaPipe) -> pinch detector
    -> teaching session -> demonstration events -> overlay -> publish image

Topics out:
    /ar_teach/image          sensor_msgs/Image                 AR view with overlay
    /teach/demonstrations    sorting_interfaces/Demonstration  reliable, latched (depth 100)
    /ar_teach/status         std_msgs/String                   current banner text, "" = none (latched)
Topics in:
    /teach/query             sorting_interfaces/Item           question item (volatile; re-sent if unanswered)
Services:
    ~/new_scene              std_srvs/Trigger                  new random scene and teaching session

Failure handling (no crash in any of these):
- No webcam: a placeholder frame with "No webcam found..." is published
  until a camera appears (the capture thread keeps retrying).
- No hand model: the banner stays up for as long as the model is missing.
- Invalid query message: rejected with a warning.
"""

from __future__ import annotations

import math
import os
import time
import uuid

import numpy as np
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from sorting_interfaces.msg import Demonstration
from sorting_interfaces.msg import Item as ItemMsg
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ar_teach.hand_tracking import HandTracker, ModelLoadError
from ar_teach.lighting import LightingConfig, lighting_message
from ar_teach.palette import BLACK_GREY
from ar_teach.pinch import PinchConfig, PinchDetector
from ar_teach.render import OverlayRenderer
from ar_teach.session import DemoEvent, MessageEvent, SessionConfig, TeachSession
from ar_teach.synthetic import ScriptedRule, SyntheticSource
from ar_teach.webcam import MSG_NO_WEBCAM, WebcamConfig, WebcamSource
from sorting_common.assets import font_path
from sorting_common.image_utils import IMAGE_QOS, cv2_to_imgmsg
from sorting_common.node_runner import run_node
from sorting_scene import Item, default_config_path, generate_scene, load_config, place_item

# Demonstrations are rare and must not be lost: reliable, and latched so a
# rule learner started later still receives the whole session.
DEMO_QOS = QoSProfile(
    depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
STATS_PERIOD_S = 5.0


def _stamp(t: float) -> Time:
    """Wall-clock seconds -> ROS Time.

    Teaching runs on real time (webcam capture times), not sim time, so wall
    time is deliberate here; nodes that must follow ``use_sim_time`` use the
    node clock instead.
    """
    sec = int(t)
    return Time(sec=sec, nanosec=int((t - sec) * 1e9))


class TeachNode(Node):
    """Runs one AR teaching session at a time."""

    def __init__(self) -> None:
        super().__init__("ar_teach")
        # --- outputs first: anything below may already report a problem ------
        self.image_pub = self.create_publisher(Image, "/ar_teach/image", IMAGE_QOS)
        self.status_pub = self.create_publisher(String, "/ar_teach/status", LATCHED)
        self.demo_pub = self.create_publisher(Demonstration, "/teach/demonstrations", DEMO_QOS)
        self.message: tuple[str, float] | None = None  # transient banner (text, time)
        self.faults: dict[str, str] = {}  # persistent banners while a fault lasts
        self._last_status: str | None = None

        p = self.declare_parameter
        self.source_kind = str(p("source", "webcam").value)  # "webcam" | "synthetic"
        scene_yaml = str(p("scene_config", "").value) or str(default_config_path())
        self.scene_seed = int(p("scene_seed", 0).value)
        self.fps = float(p("fps", 30.0).value)
        self.message_hold_s = float(p("message_hold_s", 3.0).value)
        model = str(p("model_path", "").value) or os.environ.get("SORTING_HAND_MODEL", "")
        webcam_cfg = WebcamConfig(
            index=int(p("webcam.index", 0).value),
            width=int(p("webcam.width", 640).value),
            height=int(p("webcam.height", 480).value),
            fps=self.fps,
            mirror=bool(p("webcam.mirror", True).value),
            lock_framerate=bool(p("webcam.lock_framerate", True).value),
            reopen_period_s=float(p("webcam.reopen_period_s", 2.0).value),
            max_failed_reads=int(p("webcam.max_failed_reads", 15).value),
        )
        self.pinch_cfg = PinchConfig(
            grab_ratio=float(p("pinch.grab_ratio", 0.30).value),
            release_ratio=float(p("pinch.release_ratio", 0.45).value),
            ratio_alpha=float(p("pinch.ratio_alpha", 0.6).value),
            cursor_alpha=float(p("pinch.cursor_alpha", 0.5).value),
            grab_frames=int(p("pinch.grab_frames", 2).value),
            release_frames=int(p("pinch.release_frames", 2).value),
        )
        self.session_cfg = SessionConfig(
            hover_margin_px=float(p("session.hover_margin_px", 8.0).value),
            lost_timeout_s=float(p("session.lost_timeout_s", 0.4).value),
        )
        self.lighting_cfg = LightingConfig(
            dark_mean=float(p("lighting.dark_mean", 45.0).value),
            bright_clipped_fraction=float(p("lighting.bright_clipped_fraction", 0.35).value),
        )
        min_det = float(p("tracking.min_detection_confidence", 0.5).value)
        min_track = float(p("tracking.min_tracking_confidence", 0.5).value)
        rule = ScriptedRule(
            attribute=str(p("synthetic.attribute", "color").value),
            mapping=dict(
                zip(
                    list(p("synthetic.values", ["red", "blue"]).value),
                    list(p("synthetic.boxes", ["A", "B"]).value),
                    strict=True,
                )
            ),
            default=str(p("synthetic.default_box", "C").value),
        )

        self.scene_cfg = load_config(scene_yaml)
        self.renderer = OverlayRenderer(self.scene_cfg, font_path())
        self.detector = PinchDetector(self.pinch_cfg)
        self.session_id = ""
        self.session = self._new_session()
        self.demo_index = 0
        self._query_rng = np.random.default_rng(self.scene_seed)

        self.webcam: WebcamSource | None = None
        self.synthetic: SyntheticSource | None = None
        self.tracker: HandTracker | None = None
        if self.source_kind == "synthetic":
            self.synthetic = SyntheticSource(
                self.session, rule, (webcam_cfg.width, webcam_cfg.height), self.fps
            )
        elif self.source_kind == "webcam":
            try:
                self.tracker = HandTracker(model, min_det, min_track)
            except FileNotFoundError as exc:
                self.get_logger().error(str(exc))  # keep running; the banner shows the fix
                self.faults["model"] = "Hand model missing. Run scripts/download_model.sh, then restart."
            except ModelLoadError as exc:
                self.get_logger().error(str(exc))
                self.faults["model"] = (
                    "Hand model could not be loaded. Re-run scripts/download_model.sh, then restart."
                )
            self.webcam = WebcamSource(webcam_cfg, self._log)
        else:
            raise ValueError(f"Unknown source '{self.source_kind}'. Use 'webcam' or 'synthetic'.")
        self._frame_size = (webcam_cfg.width, webcam_cfg.height)

        # Volatile: a restarted node must not get a stale question; the
        # learner re-asks unanswered questions after a timeout.
        self.create_subscription(
            ItemMsg,
            "/teach/query",
            self._on_query,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_service(Trigger, "~/new_scene", self._on_new_scene)
        self._last_seq = -1
        self._stats = {"frames": 0, "hand": 0, "work_ms": 0.0}
        self.create_timer(1.0 / self.fps, self._on_timer)
        self.create_timer(STATS_PERIOD_S, self._log_stats)
        self.get_logger().info(
            f"AR teaching started (source={self.source_kind}, scene seed {self.scene_seed}, "
            f"{len(self.session.items)} items, session {self.session_id})"
        )

    # --- helpers -------------------------------------------------------------
    def _log(self, level: str, text: str) -> None:
        """Logger callback for the webcam thread (the 'no webcam' fault itself is set in ``_on_timer``)."""
        logger = self.get_logger()
        {"info": logger.info, "warning": logger.warning, "error": logger.error}.get(level, logger.info)(text)

    def _new_session(self) -> TeachSession:
        self.session_id = uuid.uuid4().hex[:8]
        session = TeachSession(self.scene_cfg, self.session_cfg)
        for item in generate_scene(self.scene_cfg, self.scene_seed).items:
            session.add_item(item)
        return session

    def _set_message(self, text: str) -> None:
        self.message = (text, time.monotonic())

    def _banner(self, lighting: str | None) -> str | None:
        """Priority: persistent faults > recent event message > lighting."""
        if self.faults:
            return next(iter(self.faults.values()))
        if self.message and time.monotonic() - self.message[1] < self.message_hold_s:
            return self.message[0]
        return lighting

    def _publish_status(self, banner: str | None) -> None:
        text = banner or ""
        if text != self._last_status:
            self._last_status = text
            self.status_pub.publish(String(data=text))

    # --- callbacks ------------------------------------------------------------
    def _on_timer(self) -> None:
        t0 = time.perf_counter()
        now = time.time()
        frame: np.ndarray
        landmarks: np.ndarray | None = None
        real_frame = True  # False for the placeholder shown when there's no webcam
        if self.synthetic is not None:
            frame, landmarks = self.synthetic.next()
        elif self.webcam is not None:
            if self.webcam.available:
                self.faults.pop("webcam", None)
            seq, stamp, latest = self.webcam.latest()
            if latest is None or not self.webcam.available:
                if not self.webcam.available:
                    self.faults["webcam"] = MSG_NO_WEBCAM
                # Placeholder frame so the view (and the fault banner) stay visible.
                w, h = self._frame_size
                frame = np.full((h, w, 3), BLACK_GREY, np.uint8)
                real_frame = False
            elif seq == self._last_seq:
                return  # no new camera frame yet
            else:
                self._last_seq, now = seq, stamp
                frame = latest.copy()
                landmarks = self.tracker.detect(frame, now) if self.tracker else None
        else:
            return

        pinch = self.detector.update(landmarks)
        for ev in self.session.update(now, pinch):
            if isinstance(ev, DemoEvent):
                self._publish_demo(ev)
            elif isinstance(ev, MessageEvent):
                (self.get_logger().warning if ev.level == "warning" else self.get_logger().info)(ev.text)
                self._set_message(ev.text)
                self.detector.reset()
                self.session.disarm()
        lighting = lighting_message(frame, self.lighting_cfg) if real_frame else None
        banner = self._banner(lighting)
        self.renderer.draw(frame, self.session, pinch, banner)
        self._publish_status(banner)
        self.image_pub.publish(cv2_to_imgmsg(frame, _stamp(now), "ar_view"))

        self._stats["frames"] += 1
        self._stats["hand"] += landmarks is not None
        self._stats["work_ms"] += (time.perf_counter() - t0) * 1000

    def _publish_demo(self, ev: DemoEvent) -> None:
        self.demo_index += 1
        msg = Demonstration()
        msg.header.stamp = _stamp(ev.t)  # time of the drop
        msg.header.frame_id = "world"
        msg.session_id = self.session_id
        msg.scene_seed = self.scene_seed
        msg.index = self.demo_index
        it = ev.item
        msg.item = ItemMsg(id=it.id, color=it.color, size=it.size, shape=it.shape, x=it.x, y=it.y, yaw=it.yaw)
        msg.box = ev.box
        msg.source = "query" if ev.is_query else "ar"
        msg.duration_s = ev.duration_s
        self.demo_pub.publish(msg)
        self.get_logger().info(
            f"Demonstration #{msg.index}: {it.describe()} -> box {ev.box} "
            f"({ev.duration_s:.1f} s, source {msg.source})"
        )

    def _on_query(self, msg: ItemMsg) -> None:
        """Add an active-learning question item, after validating it."""
        attrs = {"color": msg.color, "size": msg.size, "shape": msg.shape}
        for name, allowed in self.scene_cfg.attribute_values.items():
            if attrs[name] not in allowed:
                self.get_logger().warning(f"Rejected query {msg.id!r}: unknown {name} '{attrs[name]}'.")
                return
        if not all(math.isfinite(v) for v in (msg.x, msg.y, msg.yaw)):
            self.get_logger().warning(f"Rejected query {msg.id!r}: x, y and yaw must be finite numbers.")
            return
        if not msg.id:
            self.get_logger().warning("Rejected query: empty id.")
            return
        if msg.id in self.session.items:
            # Normal: the learner re-sends an unanswered question; it's already on the table.
            self.get_logger().debug(f"Question {msg.id} is already on the table.")
            return
        existing = [vi.item for vi in self.session.items.values() if vi.status != "boxed"]
        item = Item(id=msg.id, x=msg.x, y=msg.y, yaw=msg.yaw, **attrs)
        r = self.scene_cfg.footprint_radius(item.size)
        free = self.scene_cfg.items_area.contains(item.x, item.y, margin=r) and all(
            np.hypot(item.x - o.x, item.y - o.y) >= r + self.scene_cfg.footprint_radius(o.size)
            for o in existing
        )
        if not free:  # (0, 0) or an occupied spot: choose a free place ourselves
            placed = place_item(self.scene_cfg, attrs, self._query_rng, existing, msg.id)
            if placed is None:
                self.get_logger().warning(
                    "No free space for the question item. Clear some items from the table."
                )
                self._set_message("No room for the next question. Sort or clear some items first.")
                return
            item = placed
        self.session.add_item(item, is_query=True)
        self.get_logger().info(f"Question item {item.id}: {item.describe()} at ({item.x:.3f}, {item.y:.3f})")
        self._set_message("Where does this item go? Pick it up and drop it in a box.")

    def _on_new_scene(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self.scene_seed += 1
        self.session = self._new_session()
        if self.synthetic is not None:
            self.synthetic.session = self.session
        self.detector.reset()
        self.demo_index = 0
        # Recreate the latched publisher so late subscribers don't receive the
        # previous session's demonstrations; session_id also tells them apart.
        self.destroy_publisher(self.demo_pub)
        self.demo_pub = self.create_publisher(Demonstration, "/teach/demonstrations", DEMO_QOS)
        resp.success, resp.message = (
            True,
            f"New scene with seed {self.scene_seed} (session {self.session_id})",
        )
        self.get_logger().info(resp.message)
        return resp

    def _log_stats(self) -> None:
        n = self._stats["frames"]
        if n == 0:
            self.get_logger().warning("No frames processed in the last 5 s. Check the webcam.")
            return
        self.get_logger().info(
            f"{n / STATS_PERIOD_S:.1f} fps, hand seen in {100 * self._stats['hand'] / n:.0f}% of frames, "
            f"{self._stats['work_ms'] / n:.1f} ms per frame"
        )
        self._stats = {"frames": 0, "hand": 0, "work_ms": 0.0}

    def destroy_node(self) -> None:
        """Release the webcam and the hand tracker."""
        if self.webcam is not None:
            self.webcam.close()
        if self.tracker is not None:
            self.tracker.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run ar_teach teach_node``."""
    run_node(TeachNode, args)


if __name__ == "__main__":
    main()
