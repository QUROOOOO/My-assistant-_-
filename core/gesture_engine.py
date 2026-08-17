import asyncio
import collections
import json
import math
import os
import threading
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from core.skills.kinematics_validator import OneEuroFilter, KinematicsValidator
from core.skills.gesture_arbitrator import GestureArbitrator

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[PIPO Vision] Downloading HandLandmarker model (~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[PIPO Vision] Model ready.")


# ---------------------------------------------------------------------------
# Per-hand tracked state with Zero-Drift Deadband & Normalized Extension Ratio
# ---------------------------------------------------------------------------
class HandState:
    def __init__(self, raw_cx, raw_cy, raw_depth, raw_pinch, pinch_pos, label, is_open, e_hand, now):
        self.filters = {
            "x":       OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "y":       OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "depth":   OneEuroFilter(min_cutoff=1.0, beta=0.005),
            "pinch":   OneEuroFilter(min_cutoff=2.0, beta=0.004),
            "pinch_x": OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "pinch_y": OneEuroFilter(min_cutoff=1.5, beta=0.008),
        }
        self.x = float(self.filters["x"](raw_cx, now))
        self.y = float(self.filters["y"](raw_cy, now))
        self.depth = float(self.filters["depth"](raw_depth, now))
        self.pinch = float(self.filters["pinch"](raw_pinch, now))
        self.pinch_x = float(self.filters["pinch_x"](pinch_pos[0], now))
        self.pinch_y = float(self.filters["pinch_y"](pinch_pos[1], now))

        self.vx = 0.0
        self.vy = 0.0
        self.prev_x = float(raw_cx)
        self.prev_y = float(raw_cy)
        self.prev_raw_x = float(raw_cx)
        self.prev_raw_y = float(raw_cy)

        self.is_pinching = bool(raw_pinch < GestureArbitrator.PINCH_ENTER_RATIO)
        self.label = str(label)
        self.is_open = bool(is_open)

        # Normalized Knuckle-to-Fingertip Extension Ratio (E_hand)
        self.e_hand = float(e_hand)
        self.e_history = collections.deque(maxlen=10) # (time, e_hand) within ~130ms
        self.e_history.append((now, e_hand))
        self.is_fist = bool(e_hand < GestureArbitrator.FIST_ENTER_THRESHOLD)
        self.bloom_triggered = False

        # 3D palm normal Euler angles
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

    def update(self, raw_cx, raw_cy, raw_depth, raw_pinch, pinch_pos, label, is_open, e_hand, dt, now,
               pitch=0.0, yaw=0.0, roll=0.0):
        # ── Zero-Drift Deadband Filtering on Stationary Hands (< 0.035) ──
        raw_vx = (raw_cx - self.prev_raw_x) / max(dt, 0.001)
        raw_vy = (raw_cy - self.prev_raw_y) / max(dt, 0.001)
        raw_speed = math.sqrt(raw_vx**2 + raw_vy**2)

        if raw_speed < 0.035:
            # Stationary hold: freeze coordinate updates to eliminate landmark jitter
            self.vx = 0.0
            self.vy = 0.0
        else:
            self.x = float(self.filters["x"](raw_cx, now))
            self.y = float(self.filters["y"](raw_cy, now))
            vx = (self.x - self.prev_x) / dt
            vy = (self.y - self.prev_y) / dt
            self.vx = float(self.vx * 0.55 + vx * 0.45)
            self.vy = float(self.vy * 0.55 + vy * 0.45)
            self.prev_x = self.x
            self.prev_y = self.y

        self.prev_raw_x = raw_cx
        self.prev_raw_y = raw_cy

        self.depth = float(self.filters["depth"](raw_depth, now))
        self.pinch = float(self.filters["pinch"](raw_pinch, now))
        self.pinch_x = float(self.filters["pinch_x"](pinch_pos[0], now))
        self.pinch_y = float(self.filters["pinch_y"](pinch_pos[1], now))

        # ── Normalized Extension Ratio & Bloom Velocity (120ms window) ──
        self.e_hand = float(e_hand)
        self.e_history.append((now, e_hand))

        while len(self.e_history) > 1 and (now - self.e_history[0][0]) > 0.130:
            self.e_history.popleft()

        dE_dt = 0.0
        if len(self.e_history) >= 2:
            dt_window = max(now - self.e_history[0][0], 0.001)
            dE_dt = (self.e_hand - self.e_history[0][1]) / dt_window

        was_fist = self.is_fist
        if self.is_fist:
            if self.e_hand > GestureArbitrator.FIST_EXIT_THRESHOLD:
                self.is_fist = False
        else:
            if self.e_hand < GestureArbitrator.FIST_ENTER_THRESHOLD:
                self.is_fist = True

        # Bloom Trigger
        self.bloom_triggered = bool(
            (was_fist or self.e_hand < 0.95) and
            (dE_dt > GestureArbitrator.BLOOM_VELOCITY_THRESHOLD) and
            (self.e_hand > 1.15)
        )

        self.label = str(label)
        self.is_open = bool(is_open)
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

        # Scale-invariant pinch hysteresis
        if self.is_pinching:
            if self.pinch > GestureArbitrator.PINCH_EXIT_RATIO:
                self.is_pinching = False
        else:
            if self.pinch < GestureArbitrator.PINCH_ENTER_RATIO:
                self.is_pinching = True


class GestureEngine:
    def __init__(self):
        ensure_model()
        self.connected_clients = set()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.40,
            min_hand_presence_confidence=0.40,
            min_tracking_confidence=0.40
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.running = True

        # Handedness-keyed tracking dictionaries
        self.hand_states = {}           # label -> HandState
        self.persistence_counters = {}  # label -> int

        # Dual-hand 1€ filters for rock-solid pinch zoom
        self.dual_dist_filter = OneEuroFilter(min_cutoff=1.2, beta=0.006)
        self.dual_cx_filter   = OneEuroFilter(min_cutoff=1.5, beta=0.008)
        self.dual_cy_filter   = OneEuroFilter(min_cutoff=1.5, beta=0.008)
        self.dual_angle_filter = OneEuroFilter(min_cutoff=1.0, beta=0.005)

        # 60 FPS Event-Driven Stream Buffer
        self.latest_jpeg = None
        self.stream_viewers = 0
        self._jpeg_lock = threading.Lock()
        self._viewer_events = set()
        self._async_loop = None

        self.last_frame_time = time.time()
        self.latest_state = {
            "hands": [],
            "arbitrated_state": "IDLE",
            "two_hand_dist": 0.0,
            "dual_angle": 0.0,
            "dual_pinch": False,
            "dual_pinch_center": {"x": 0.5, "y": 0.5},
            "bloom": False,
            "compress": False,
            "slap_impulse": {"active": False, "vx": 0.0, "vy": 0.0}
        }

    async def register(self, ws):
        self.connected_clients.add(ws)
        try:
            while True:
                await asyncio.sleep(1)
        except Exception:
            pass
        finally:
            self.connected_clients.discard(ws)

    async def broadcast(self):
        if self.connected_clients:
            msg = json.dumps(self.latest_state)
            dead = set()
            for client in list(self.connected_clients):
                try:
                    await client.send_text(msg)
                except Exception:
                    dead.add(client)
            self.connected_clients -= dead

    def _notify_stream_viewers(self):
        for ev in list(self._viewer_events):
            ev.set()

    async def generate_mjpeg_stream(self):
        """Asynchronous non-blocking 60 FPS MJPEG streaming generator."""
        self.stream_viewers += 1
        ev = asyncio.Event()
        self._viewer_events.add(ev)
        last_sent = None
        try:
            while self.running:
                try:
                    await asyncio.wait_for(ev.wait(), timeout=0.033)
                    ev.clear()
                except asyncio.TimeoutError:
                    pass

                jpeg_data = None
                with self._jpeg_lock:
                    if self.latest_jpeg is not None and self.latest_jpeg is not last_sent:
                        jpeg_data = self.latest_jpeg
                        last_sent = jpeg_data
                if jpeg_data is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_data + b"\r\n"
                    )
        except (asyncio.CancelledError, GeneratorExit, Exception):
            pass
        finally:
            self._viewer_events.discard(ev)
            self.stream_viewers = max(0, self.stream_viewers - 1)

    async def generate_mjpeg(self):
        async for chunk in self.generate_mjpeg_stream():
            yield chunk

    def process_frame(self, frame):
        now = time.time()
        dt = max(now - self.last_frame_time, 0.001)
        self.last_frame_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)

        h, w, _ = frame.shape
        hands_data = []
        slap_active = False
        slap_vx, slap_vy = 0.0, 0.0
        global_bloom = False
        global_compress = False

        detected_labels = set()

        if res.hand_landmarks:
            for idx, lms in enumerate(res.hand_landmarks):
                label = "Right"
                if res.handedness and idx < len(res.handedness):
                    label = res.handedness[idx][0].category_name

                if label in detected_labels:
                    label = f"{label}_2"

                detected_labels.add(label)
                self.persistence_counters[label] = 4

                # ── Render 21-node skeletal mesh on frame ────────────────
                for start_idx, end_idx in HAND_CONNECTIONS:
                    pt1 = (int(lms[start_idx].x * w), int(lms[start_idx].y * h))
                    pt2 = (int(lms[end_idx].x * w), int(lms[end_idx].y * h))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

                for lm in lms:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

                wrist_pt = (int(lms[0].x * w), int(lms[0].y * h) + 20)
                cv2.putText(frame, label.upper(), wrist_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # ── Kinematic metrics via KinematicsValidator ────────────
                raw_cx = (lms[0].x + lms[9].x) / 2.0
                raw_cy = (lms[0].y + lms[9].y) / 2.0

                knuckle_ref = KinematicsValidator.dist_3d(lms[9], lms[0])
                raw_depth = knuckle_ref * 3.5

                pinch_ratio = KinematicsValidator.calculate_pinch_ratio(lms[4], lms[8], knuckle_ref)
                pinch_center = ((lms[4].x + lms[8].x) / 2.0, (lms[4].y + lms[8].y) / 2.0)

                tips = [lms[8], lms[12], lms[16], lms[20]]
                e_hand = KinematicsValidator.calculate_extension_ratio(lms[0], lms[9], tips)
                is_open = e_hand > 1.15
                pitch, yaw, roll = KinematicsValidator.calculate_palm_euler(lms[0], lms[9], lms[5], lms[17])

                # ── Handedness-keyed update with deadband filter ─────────
                if label not in self.hand_states:
                    self.hand_states[label] = HandState(
                        raw_cx, raw_cy, raw_depth, pinch_ratio, pinch_center,
                        label, is_open, e_hand, now
                    )
                    self.hand_states[label].pitch = pitch
                    self.hand_states[label].yaw = yaw
                    self.hand_states[label].roll = roll
                else:
                    self.hand_states[label].update(
                        raw_cx, raw_cy, raw_depth, pinch_ratio, pinch_center,
                        label, is_open, e_hand, dt, now,
                        pitch=pitch, yaw=yaw, roll=roll
                    )

                sm = self.hand_states[label]
                speed = math.sqrt(sm.vx**2 + sm.vy**2)

                if speed > 1.8 and is_open:
                    slap_active = True
                    slap_vx = sm.vx
                    slap_vy = sm.vy

                if sm.bloom_triggered:
                    global_bloom = True
                if sm.is_fist:
                    global_compress = True

        # ── Persistence cleanup ──────────────────────────────────────────
        all_tracked = list(self.hand_states.keys())
        for label in all_tracked:
            if label not in detected_labels:
                cnt = self.persistence_counters.get(label, 0) - 1
                self.persistence_counters[label] = cnt
                if cnt <= 0:
                    del self.hand_states[label]
                    self.persistence_counters.pop(label, None)

        # ── Construct Hands Telemetry Payload ────────────────────────────
        for label, sm in self.hand_states.items():
            hands_data.append({
                "id": label,
                "label": sm.label,
                "palm_x": sm.x,
                "palm_y": sm.y,
                "pinch_x": sm.pinch_x,
                "pinch_y": sm.pinch_y,
                "depth_scale": min(max(sm.depth, 0.5), 2.2),
                "pinch_dist": sm.pinch,
                "is_pinching": sm.is_pinching,
                "is_open": sm.is_open,
                "is_fist": sm.is_fist,
                "e_hand": round(sm.e_hand, 3),
                "pitch": round(sm.pitch, 4),
                "yaw": round(sm.yaw, 4),
                "roll": round(sm.roll, 4),
            })

        # ── Dual-Hand Pinch Zoom & Scale (Keyed by Left & Right Pinches) ─
        two_hand_dist = 0.0
        dual_angle = 0.0
        dual_pinch = False
        dual_pinch_center = {"x": 0.5, "y": 0.5}

        pinching_hands = [h for h in hands_data if h["is_pinching"]]
        if len(pinching_hands) >= 2:
            p1, p2 = pinching_hands[0], pinching_hands[1]
            raw_dx = p2["pinch_x"] - p1["pinch_x"]
            raw_dy = p2["pinch_y"] - p1["pinch_y"]
            raw_dist = math.sqrt(raw_dx**2 + raw_dy**2)
            raw_angle = math.atan2(raw_dy, raw_dx)
            raw_cx = (p1["pinch_x"] + p2["pinch_x"]) / 2.0
            raw_cy = (p1["pinch_y"] + p2["pinch_y"]) / 2.0

            two_hand_dist = self.dual_dist_filter(raw_dist, now)
            dual_angle = self.dual_angle_filter(raw_angle, now)
            dual_pinch_center = {
                "x": self.dual_cx_filter(raw_cx, now),
                "y": self.dual_cy_filter(raw_cy, now)
            }
            dual_pinch = True
        elif len(hands_data) == 2:
            p1, p2 = hands_data[0], hands_data[1]
            raw_dx = p2["palm_x"] - p1["palm_x"]
            raw_dy = p2["palm_y"] - p1["palm_y"]
            raw_dist = math.sqrt(raw_dx**2 + raw_dy**2)
            raw_angle = math.atan2(raw_dy, raw_dx)
            raw_cx = (p1["palm_x"] + p2["palm_x"]) / 2.0
            raw_cy = (p1["palm_y"] + p2["palm_y"]) / 2.0

            two_hand_dist = self.dual_dist_filter(raw_dist, now)
            dual_angle = self.dual_angle_filter(raw_angle, now)
            dual_pinch_center = {
                "x": self.dual_cx_filter(raw_cx, now),
                "y": self.dual_cy_filter(raw_cy, now)
            }

        arbitrated_state = GestureArbitrator.arbitrate_state(
            hands=hands_data,
            dual_pinch=dual_pinch,
            global_bloom=global_bloom,
            global_compress=global_compress,
            slap_active=slap_active
        )

        self.latest_state = {
            "hands": hands_data,
            "arbitrated_state": arbitrated_state,
            "two_hand_dist": two_hand_dist,
            "dual_angle": round(dual_angle, 4),
            "dual_pinch": dual_pinch,
            "dual_pinch_center": dual_pinch_center,
            "bloom": global_bloom,
            "compress": global_compress,
            "slap_impulse": {"active": slap_active, "vx": slap_vx, "vy": slap_vy}
        }

        # ── 60 FPS Conditional Live Feed Push (420x236 @ Quality 70) ────
        if self.stream_viewers > 0:
            preview = cv2.resize(frame, (420, 236), interpolation=cv2.INTER_LINEAR)
            ret, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                with self._jpeg_lock:
                    self.latest_jpeg = jpeg.tobytes()
                if self._async_loop and not self._async_loop.is_closed():
                    self._async_loop.call_soon_threadsafe(self._notify_stream_viewers)

        return frame

    def run_capture(self, loop):
        self._async_loop = loop

        def open_camera():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap = open_camera()
        consecutive_drops = 0

        while self.running:
            if not cap.isOpened():
                time.sleep(0.5)
                cap = open_camera()
                continue

            ret, frame = cap.read()
            if not ret:
                consecutive_drops += 1
                if consecutive_drops >= 5:
                    print("[PIPO Vision] Watchdog: 5 dropped frames. Recovering camera pipeline...")
                    cap.release()
                    time.sleep(0.2)
                    cap = open_camera()
                    consecutive_drops = 0
                time.sleep(0.005)
                continue

            consecutive_drops = 0
            frame = cv2.flip(frame, 1)
            self.process_frame(frame)

            asyncio.run_coroutine_threadsafe(self.broadcast(), loop)

        cap.release()
