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

from core.skills.kinematics_validator import OneEuroFilter
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
# Per-hand tracked state with Strict 2-Phase Bloom Gating & Zero-Drift Deadband
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

        # Strict 2-Phase Fist Charge & Bloom State
        self.e_hand = float(e_hand)
        self.e_history = collections.deque(maxlen=12) # (time, e_hand) within ~130ms
        self.e_history.append((now, e_hand))

        self.fist_charge_counter = 0
        self.fist_charged = False
        self.is_fist = False
        self.bloom_triggered = False

        # 3D palm normal Euler angles
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

    def update(self, raw_cx, raw_cy, raw_depth, raw_pinch, pinch_pos, label, is_open, e_hand, thumb_tucked, dt, now,
               pitch=0.0, yaw=0.0, roll=0.0):
        # ── Zero-Drift Deadband Filtering on Stationary Hands (< 0.035) ──
        raw_vx = (raw_cx - self.prev_raw_x) / max(dt, 0.001)
        raw_vy = (raw_cy - self.prev_raw_y) / max(dt, 0.001)
        raw_speed = math.sqrt(raw_vx**2 + raw_vy**2)

        if raw_speed < 0.035:
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

        # ── Strict 2-Phase Bloom Sequence Gating ────────────────────────
        self.e_hand = float(e_hand)
        self.e_history.append((now, e_hand))

        while len(self.e_history) > 1 and (now - self.e_history[0][0]) > 0.130:
            self.e_history.popleft()

        dE_dt = 0.0
        if len(self.e_history) >= 2:
            dt_window = max(now - self.e_history[0][0], 0.001)
            dE_dt = (self.e_hand - self.e_history[0][1]) / dt_window

        # Phase 1: Charge Confirmation (E_hand < 0.72 AND thumb tucked for >= 5 frames)
        if self.e_hand < GestureArbitrator.FIST_CHARGE_THRESHOLD and thumb_tucked:
            self.fist_charge_counter += 1
            if self.fist_charge_counter >= GestureArbitrator.FIST_CHARGE_MIN_FRAMES:
                self.fist_charged = True
                self.is_fist = True
        else:
            if not self.fist_charged:
                self.fist_charge_counter = max(0, self.fist_charge_counter - 1)
                self.is_fist = False

        # Phase 2: Explosive Snap Trigger vs Safe Slow-Open Reversion
        self.bloom_triggered = False
        if self.fist_charged:
            # Explosive Snap Trigger
            if dE_dt > GestureArbitrator.BLOOM_VELOCITY_THRESHOLD and self.e_hand > 1.15:
                self.bloom_triggered = True
                self.fist_charged = False
                self.fist_charge_counter = 0
                self.is_fist = False
            # Safe Slow-Open Reversion
            elif self.e_hand > GestureArbitrator.FIST_EXIT_THRESHOLD and dE_dt < GestureArbitrator.SLOW_OPEN_VELOCITY_LIMIT:
                self.fist_charged = False
                self.fist_charge_counter = 0
                self.is_fist = False

        self.label = str(label)
        self.is_open = bool(is_open)
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

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

        # Dual-hand 1€ filters
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

        # Dedicated Ingestion Thread Buffer (maxlen=1 prevents queue lag)
        self._frame_buffer = collections.deque(maxlen=1)

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
                await ws.receive_text()
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

                # ── Vectorized NumPy Landmark Array Conversion ───────────
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                # ── Render 21-node skeletal mesh on frame ────────────────
                pts_2d = (pts[:, :2] * np.array([w, h], dtype=np.float32)).astype(np.int32)
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts_2d[start_idx]), tuple(pts_2d[end_idx]), (255, 255, 0), 2, cv2.LINE_AA)

                for pt in pts_2d:
                    cv2.circle(frame, tuple(pt), 4, (0, 255, 255), -1, cv2.LINE_AA)

                wrist_pt = (pts_2d[0][0], pts_2d[0][1] + 20)
                cv2.putText(frame, label.upper(), wrist_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # ── SIMD Vectorized Kinematic Metrics ────────────────────
                wrist = pts[0]
                middle_mcp = pts[9]
                index_mcp = pts[5]
                pinky_mcp = pts[17]

                raw_cx = float((wrist[0] + middle_mcp[0]) / 2.0)
                raw_cy = float((wrist[1] + middle_mcp[1]) / 2.0)

                knuckle_ref = float(np.linalg.norm(middle_mcp - wrist))
                raw_depth = knuckle_ref * 3.5

                thumb_tip = pts[4]
                index_tip = pts[8]
                tip_dist = float(np.linalg.norm(thumb_tip - index_tip))
                pinch_ratio = tip_dist / knuckle_ref if knuckle_ref > 1e-6 else 1.0
                pinch_center = (float((thumb_tip[0] + index_tip[0]) / 2.0),
                                float((thumb_tip[1] + index_tip[1]) / 2.0))

                # Tips: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
                four_tips = pts[[8, 12, 16, 20]]
                dists_to_wrist = np.linalg.norm(four_tips - wrist, axis=1)
                e_hand = float(np.sum(dists_to_wrist) / (4.0 * max(knuckle_ref, 1e-6)))

                # Thumb tucked check: distance from thumb tip to wrist vs knuckle ref
                thumb_wrist_dist = float(np.linalg.norm(thumb_tip - wrist))
                thumb_tucked = bool(thumb_wrist_dist / max(knuckle_ref, 1e-6) < 0.92)

                is_open = e_hand > 1.15

                # 3D Palm Euler Vectorization
                v1 = middle_mcp - wrist
                v2 = pinky_mcp - index_mcp
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                normal = normal / norm_len if norm_len > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

                pitch = float(math.asin(max(-1.0, min(1.0, -normal[1]))))
                yaw   = float(math.atan2(normal[0], normal[2]))
                roll  = float(math.atan2(v2[1], math.sqrt(v2[0]**2 + v2[2]**2)))

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
                        label, is_open, e_hand, thumb_tucked, dt, now,
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
                "fist_charged": sm.fist_charged,
                "e_hand": round(sm.e_hand, 3),
                "pitch": round(sm.pitch, 4),
                "yaw": round(sm.yaw, 4),
                "roll": round(sm.roll, 4),
            })

        # ── Dual-Hand Pinch Zoom & Scale ─────────────────────────────────
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

        # Dedicated Background Ingestion Thread
        def ingestion_worker():
            nonlocal cap, consecutive_drops
            while self.running:
                if not cap.isOpened():
                    time.sleep(0.5)
                    cap = open_camera()
                    continue

                ret, raw_frame = cap.read()
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
                self._frame_buffer.append(raw_frame)

        ingestion_thread = threading.Thread(target=ingestion_worker, daemon=True)
        ingestion_thread.start()

        # Processing Loop (reads latest frame from single-slot buffer)
        while self.running:
            if self._frame_buffer:
                raw_frame = self._frame_buffer.pop()
                flipped = cv2.flip(raw_frame, 1)
                self.process_frame(flipped)
                asyncio.run_coroutine_threadsafe(self.broadcast(), loop)
            else:
                time.sleep(0.002)

        cap.release()
