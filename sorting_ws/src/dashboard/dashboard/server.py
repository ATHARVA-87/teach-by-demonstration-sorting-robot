"""PARKED (old gesture project, ported in M6).

dashboard_node: serves the operator console at http://localhost:8080.

Two halves in one process:
  * ROS side (a background thread): subscribes to every topic the console
    shows, keeps rolling metrics, and calls the /robot/estop service.
  * Web side (main thread): FastAPI + uvicorn. The page opens one WebSocket
    at /ws and receives:
      - binary messages: JPEG frames of /camera/annotated
      - JSON {"type": "config", ...}     once, on connect
      - JSON {"type": "telemetry", ...}  ~20 times per second
      - JSON {"type": "event", ...}      when something happens (log entry)
    and sends JSON {"type": "estop", "engage": true|false}.

Parameters: host, port, video_fps, jpeg_quality, cmd_vel_topic, pose_topic,
            world_size, static_dir
"""
import asyncio
import collections
import json
import os
import threading
import time

import cv2
import rclpy
import uvicorn
from ament_index_python.packages import get_package_share_directory
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from geometry_msgs.msg import Point, Twist
from rcl_interfaces.msg import Log
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Header, String
from std_srvs.srv import SetBool
from turtlesim_msgs.msg import Pose

from gesture_robot.image_utils import IMAGE_QOS, imgmsg_to_cv2

TELEMETRY_PERIOD_S = 0.05     # 20 Hz to the browser
LIVENESS_TIMEOUT_S = 1.0      # a node counts as "down" after 1 s of silence
STABLE_GESTURE_FRAMES = 3     # log a gesture change only once it holds 3 frames
OUR_NODES = {"camera_node", "hand_tracker_node", "controller_node", "turtlesim"}

STATE_EVENTS = {
    "ACTIVE": ("info", "Robot active. Following your hand."),
    "STOPPED": ("info", "Robot stopped."),
    "NO_HAND": ("info", "No hand detected. Robot stopped."),
    "ESTOP": ("warning", "Emergency stop engaged. Robot locked."),
}
GESTURE_NAMES = {"OPEN_PALM": "open palm", "FIST": "fist", "UNKNOWN": "unclear hand pose", "NONE": "no hand"}


class Rolling:
    """Fixed-size window of recent samples with a few summary statistics."""

    def __init__(self, size: int):
        self.values = collections.deque(maxlen=size)

    def add(self, v: float):
        self.values.append(v)

    def mean(self):
        return sum(self.values) / len(self.values) if self.values else None

    def p95(self):
        if not self.values:
            return None
        ordered = sorted(self.values)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


class RateMeter:
    """Messages per second over the last `window_s` seconds."""

    def __init__(self, window_s: float = 2.0):
        self.window_s = window_s
        self.times = collections.deque()

    def tick(self, now: float):
        self.times.append(now)

    def rate(self, now: float) -> float:
        while self.times and now - self.times[0] > self.window_s:
            self.times.popleft()
        return len(self.times) / self.window_s

    def last(self):
        return self.times[-1] if self.times else None


class DashboardRos(Node):
    """ROS side: collects everything the console displays."""

    def __init__(self):
        super().__init__("dashboard_node")
        p = self.declare_parameter
        self.host = p("host", "127.0.0.1").value      # localhost only: the page can stop the robot
        self.port = int(p("port", 8080).value)
        self.video_fps = float(p("video_fps", 30.0).value)
        self.jpeg_quality = int(p("jpeg_quality", 70).value)
        cmd_vel_topic = p("cmd_vel_topic", "/turtle1/cmd_vel").value
        pose_topic = p("pose_topic", "/turtle1/pose").value
        self.world_size = float(p("world_size", 11.0889).value)  # turtlesim arena, metres
        default_static = os.path.join(get_package_share_directory("gesture_robot"), "dashboard")
        self.static_dir = p("static_dir", default_static).value

        self.lock = threading.Lock()   # ROS thread writes, web thread reads
        self.ws_clients = 0               # open WebSockets (skip JPEG work when 0)

        # Latest values.
        self.state = None
        self.gesture = "NONE"
        self.hand = (0.0, 0.0)
        self.cmd = (0.0, 0.0)
        self.pose = None
        self.estop = False
        self.limits = {"max_linear": 2.0, "max_angular": 2.0, "dead_zone": 0.15}

        # Video frame (JPEG bytes) and a counter so senders know when it changed.
        self.jpeg = None
        self.jpeg_seq = 0
        self._last_encode = 0.0

        # Metrics. The ROS thread appends and the web thread reads; deque
        # append/popleft and list(deque) are atomic under the GIL, so these
        # need no lock.
        self.camera_rate = RateMeter()
        self.tracker_rate = RateMeter()
        self.controller_rate = RateMeter()
        self.robot_rate = RateMeter()
        self.detections = Rolling(60)  # 1 = hand seen, 0 = not; last ~2 s
        self.latency = Rolling(90)

        # Event log: kept here so a page that connects late sees history.
        self.events = collections.deque(maxlen=200)
        self.event_seq = 0
        self._gesture_candidate, self._gesture_count, self._logged_gesture = None, 0, None
        self._alive = {"camera": None, "tracker": None, "controller": None, "robot": None}

        # Trail of the robot's path (only points that moved > 2 cm).
        self.trail = collections.deque(maxlen=4000)

        # --- Subscriptions ---------------------------------------------------
        # raw=True: receive the serialized bytes without decoding the 900 KB
        # image; we only need to count camera frames.
        self.create_subscription(Image, "/camera/image_raw", self._on_raw_image, IMAGE_QOS, raw=True)
        self.create_subscription(Image, "/camera/annotated", self._on_annotated, IMAGE_QOS)
        self.create_subscription(String, "/hand/gesture", self._on_gesture, 10)
        self.create_subscription(Point, "/hand/position", self._on_position, 10)
        self.create_subscription(Header, "/hand/stamp", self._on_stamp, 10)
        self.create_subscription(Float64, "/metrics/latency_ms", self._on_latency, 10)
        self.create_subscription(Twist, cmd_vel_topic, self._on_cmd, 10)
        self.create_subscription(Pose, pose_topic, self._on_pose, 10)
        self.create_subscription(
            String, "/robot/state", self._on_state,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(Log, "/rosout", self._on_rosout, 50)

        self.estop_client = self.create_client(SetBool, "/robot/estop")

        # Read the controller's limits so the joystick draws the real dead zone.
        self._param_client = AsyncParameterClient(self, "controller_node")
        self._param_future = None
        self.create_timer(1.0, self._fetch_controller_params)
        self.create_timer(0.25, self._check_liveness)

    # --- event log ------------------------------------------------------------
    def add_event(self, level: str, text: str):
        with self.lock:
            # The same message can arrive twice (the e-stop state change and
            # the controller's own log line) or repeat (a node warning every
            # 5 s). Skip text already logged in the last 30 s.
            now = time.time()
            if any(e["text"] == text and now - e["t"] < 30.0 for e in list(self.events)[-10:]):
                return
            self.event_seq += 1
            self.events.append({"type": "event", "id": self.event_seq, "t": time.time(),
                                "level": level, "text": text})

    # --- callbacks (ROS thread) -------------------------------------------------
    def _on_raw_image(self, _msg: bytes):
        self.camera_rate.tick(time.monotonic())

    def _on_annotated(self, msg: Image):
        if self.ws_clients == 0:
            return
        now = time.monotonic()
        # Throttle to video_fps. The 0.8 tolerance stops frame-timing jitter
        # from dropping every other frame when video_fps == camera fps.
        if now - self._last_encode < 0.8 / self.video_fps:
            return
        self._last_encode = now
        ok, buf = cv2.imencode(".jpg", imgmsg_to_cv2(msg), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                self.jpeg_seq += 1

    def _on_gesture(self, msg: String):
        self.gesture = msg.data
        self.detections.add(0 if msg.data == "NONE" else 1)
        # Log only gestures that hold for a few frames, not one-frame flickers.
        if msg.data == self._gesture_candidate:
            self._gesture_count += 1
        else:
            self._gesture_candidate, self._gesture_count = msg.data, 1
        if self._gesture_count == STABLE_GESTURE_FRAMES and msg.data != self._logged_gesture:
            self._logged_gesture = msg.data
            self.add_event("info", f"Gesture: {GESTURE_NAMES.get(msg.data, msg.data)}.")

    def _on_position(self, msg: Point):
        self.hand = (msg.x, msg.y)

    def _on_stamp(self, _msg: Header):
        self.tracker_rate.tick(time.monotonic())

    def _on_latency(self, msg: Float64):
        self.latency.add(msg.data)

    def _on_cmd(self, msg: Twist):
        self.cmd = (msg.linear.x, msg.angular.z)
        self.controller_rate.tick(time.monotonic())

    def _on_pose(self, msg: Pose):
        self.robot_rate.tick(time.monotonic())
        self.pose = (msg.x, msg.y, msg.theta)
        with self.lock:
            if not self.trail or (msg.x - self.trail[-1][0]) ** 2 + (msg.y - self.trail[-1][1]) ** 2 > 0.0004:
                self.trail.append((round(msg.x, 3), round(msg.y, 3)))

    def _on_state(self, msg: String):
        if msg.data == self.state:
            return
        self.state = msg.data
        self.estop = msg.data == "ESTOP"
        level, text = STATE_EVENTS.get(msg.data, ("info", f"Robot state: {msg.data}."))
        self.add_event(level, text)

    def _on_rosout(self, msg: Log):
        # Surface warnings and errors from our nodes (e.g. "Camera not found...").
        if msg.level >= Log.WARN and msg.name in OUR_NODES:
            self.add_event("error" if msg.level >= Log.ERROR else "warning", msg.msg)

    # --- timers -------------------------------------------------------------------
    def _check_liveness(self):
        """Log when a stage of the pipeline goes silent or comes back."""
        now = time.monotonic()
        stages = {
            "camera": (self.camera_rate, "Camera stream"),
            "tracker": (self.tracker_rate, "Hand tracker"),
            "controller": (self.controller_rate, "Controller"),
            "robot": (self.robot_rate, "Robot simulator"),
        }
        for key, (meter, label) in stages.items():
            last = meter.last()
            alive = last is not None and now - last < LIVENESS_TIMEOUT_S
            previous = self._alive[key]  # None = not seen yet
            if alive and previous is not True:
                self.add_event("info", f"{label} connected.")
            elif not alive and previous is True:
                self.add_event("warning", f"{label} stopped sending data. Check that its node is running.")
            self._alive[key] = alive

    def _fetch_controller_params(self):
        if self._param_future is None:
            if self._param_client.wait_for_services(timeout_sec=0.0):
                self._param_future = self._param_client.get_parameters(
                    ["max_linear", "max_angular", "dead_zone"]
                )
        elif self._param_future.done():
            result = self._param_future.result()
            if result is not None and len(result.values) == 3:
                v = [x.double_value for x in result.values]
                self.limits = {"max_linear": v[0], "max_angular": v[1], "dead_zone": v[2]}
            self._param_future = None  # refresh every second (params can change at runtime)

    # --- snapshot for the web side --------------------------------------------------
    def config(self) -> dict:
        with self.lock:
            return {
                "type": "config",
                "world_size": self.world_size,
                "trail": list(self.trail),
                "events": list(self.events),
            }

    def telemetry(self) -> dict:
        now = time.monotonic()
        lat_last = self.latency.values[-1] if self.latency.values else None
        det = self.detections.mean()
        return {
            "type": "telemetry",
            "state": self.state,
            "estop": self.estop,
            "gesture": self.gesture,
            "hand": {"x": self.hand[0], "y": self.hand[1]},
            "cmd": {"linear": self.cmd[0], "angular": self.cmd[1]},
            "limits": self.limits,
            "pose": None if self.pose is None else {"x": self.pose[0], "y": self.pose[1], "theta": self.pose[2]},
            "nodes": {k: bool(v) for k, v in self._alive.items()},
            "metrics": {
                "camera_fps": self.camera_rate.rate(now),
                "tracker_fps": self.tracker_rate.rate(now),
                "detection_rate": det,
                "latency_ms": {"last": lat_last, "avg": self.latency.mean(), "p95": self.latency.p95()},
            },
        }

    def events_after(self, last_id: int) -> list:
        with self.lock:
            return [e for e in self.events if e["id"] > last_id]

    async def set_estop(self, engage: bool) -> tuple[bool, str]:
        """Call /robot/estop from the web thread and wait for the answer."""
        if not self.estop_client.service_is_ready():
            return False, "Controller not reachable. Check that controller_node is running."
        future = self.estop_client.call_async(SetBool.Request(data=engage))
        deadline = time.monotonic() + 2.0
        while not future.done():  # the ROS thread completes it
            if time.monotonic() > deadline:
                return False, "Controller did not answer within 2 s."
            await asyncio.sleep(0.01)
        r = future.result()
        return r.success, r.message


def make_app(ros: DashboardRos) -> FastAPI:
    app = FastAPI(title="Gesture robot console")
    # Bundled fonts, so the console looks right without internet access.
    fonts = os.path.join(get_package_share_directory("gesture_robot"), "assets", "fonts")
    if os.path.isdir(fonts):
        # follow_symlink: colcon --symlink-install installs these as symlinks.
        app.mount("/fonts", StaticFiles(directory=fonts, follow_symlink=True), name="fonts")

    @app.get("/")
    def index():
        page = os.path.join(ros.static_dir, "index.html")
        if not os.path.isfile(page):
            return PlainTextResponse(f"Dashboard page not found at {page}. Rebuild with colcon.", 503)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        ros.ws_clients += 1
        sender = None
        try:
            await ws.send_text(json.dumps(ros.config()))
            last_event = ros.event_seq
            sender = asyncio.create_task(_send_loop(ws, last_event))
            while True:  # receive commands from the page
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "estop":
                    ok, text = await ros.set_estop(bool(msg.get("engage")))
                    if not ok:
                        ros.add_event("error", text)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            ros.ws_clients -= 1
            if sender is not None:
                sender.cancel()

    async def _send_loop(ws: WebSocket, last_event: int):
        last_seq = -1
        next_telemetry = 0.0
        while True:
            if ros.jpeg_seq != last_seq and ros.jpeg is not None:
                last_seq = ros.jpeg_seq
                await ws.send_bytes(ros.jpeg)
            now = time.monotonic()
            if now >= next_telemetry:
                next_telemetry = now + TELEMETRY_PERIOD_S
                await ws.send_text(json.dumps(ros.telemetry()))
                for event in ros.events_after(last_event):
                    await ws.send_text(json.dumps(event))
                    last_event = event["id"]
            await asyncio.sleep(0.01)

    return app


def main(args=None):
    # uvicorn owns Ctrl+C here: rclpy must not install its own signal handler.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    ros = DashboardRos()
    executor = SingleThreadedExecutor()
    executor.add_node(ros)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    server = uvicorn.Server(uvicorn.Config(make_app(ros), host=ros.host, port=ros.port, log_level="warning"))
    ros.get_logger().info(f"Operator console at http://localhost:{ros.port}")
    try:
        server.run()  # blocks until Ctrl+C
    except KeyboardInterrupt:
        pass  # uvicorn re-raises Ctrl+C after shutting down cleanly
    except SystemExit:  # uvicorn exits this way when the port is taken
        ros.get_logger().error(
            f"Could not start the web server on port {ros.port}. Another app may be using it. "
            "Stop that app or set the port parameter (e.g. -p port:=8081)."
        )
    finally:
        executor.shutdown()
        ros.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
