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
# Per-Hand Tracked State with Continuous Flexion & Isolated Snap Machine
# ---------------------------------------------------------------------------
class HandState:
    def __init__(self, raw_cx, raw_cy, raw_depth, raw_pinch, pinch_pos, label, raw_flexion, now):
        self.filters = {
            "x":       OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "y":       OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "depth":   OneEuroFilter(min_cutoff=1.0, beta=0.005),
            "pinch":   OneEuroFilter(min_cutoff=2.0, beta=0.004),
            "pinch_x": OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "pinch_y": OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "flexion": OneEuroFilter(min_cutoff=1.5, beta=0.008),
        }
        self.x = float(self.filters["x"](raw_cx, now))
        self.y = float(self.filters["y"](raw_cy, now))
        self.depth = float(self.filters["depth"](raw_depth, now))
        self.pinch = float(self.filters["pinch"](raw_pinch, now))
        self.pinch_x = float(self.filters["pinch_x"](pinch_pos[0], now))
        self.pinch_y = float(self.filters["pinch_y"](pinch_pos[1], now))
        self.flexion = float(self.filters["flexion"](raw_flexion, now))

        self.vx = 0.0
        self.vy = 0.0
        self.prev_x = float(raw_cx)
        self.prev_y = float(raw_cy)
        self.prev_raw_x = float(raw_cx)
        self.prev_raw_y = float(raw_cy)

        self.is_pinching = bool(raw_pinch < GestureArbitrator.PINCH_ENTER_RATIO)
        self.label = str(label)
        self.is_open = bool(raw_flexion >= GestureArbitrator.FIST_OPEN_THRESHOLD)
        self.is_fist = bool(raw_flexion <= GestureArbitrator.FIST_TIGHT_THRESHOLD)

        # ── Continuous Flexion Tracking & Strict Bloom ───────────────────
        self.flexion_history = collections.deque(maxlen=20) # (time, flexion)
        self.flexion_history.append((now, self.flexion))
        self.bloom_triggered = False

        # ── Biomechanical Snap Tracker with Fist Isolation ───────────────
        self.snap_prime_counter = 0
        self.snap_primed = False
        self.snap_primed_time = 0.0
        self.snap_primed_middle_y = 0.0
        self.prev_snap_dist = 1.0
        self.c_vision = 0.0
        self.snap_triggered = False

        # 3D Palm Normal Euler angles
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

    def update(self, raw_cx, raw_cy, raw_depth, raw_pinch, pinch_pos, label,
               norm_tip_dists, norm_thumb_middle, middle_tip_y, dt, now,
               pitch=0.0, yaw=0.0, roll=0.0):
        # ── Zero-Drift Deadband (< 0.03) ─────────────────────────────────
        raw_vx = (raw_cx - self.prev_raw_x) / max(dt, 0.001)
        raw_vy = (raw_cy - self.prev_raw_y) / max(dt, 0.001)
        raw_speed = math.sqrt(raw_vx**2 + raw_vy**2)

        if raw_speed < GestureArbitrator.VELOCITY_DEADBAND:
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

        # ── Continuous Knuckle Extension Telemetry (1€ Filtered) ─────────
        raw_flexion = float(np.mean(norm_tip_dists))
        self.flexion = float(self.filters["flexion"](raw_flexion, now))
        self.flexion_history.append((now, self.flexion))

        # Maintain 300ms history window
        while len(self.flexion_history) > 1 and (now - self.flexion_history[0][0]) > 0.300:
            self.flexion_history.popleft()

        self.is_open = bool(self.flexion >= 1.15)
        self.is_fist = bool(self.flexion <= 0.65)

        # ── Strict Bloom: Drop below 0.50 (tight fist) -> Explosive burst (dE/dt > 3.8)
        self.bloom_triggered = False
        min_flex_in_window = min(f for t, f in self.flexion_history)
        if min_flex_in_window <= GestureArbitrator.FIST_TIGHT_THRESHOLD and self.flexion >= 1.05:
            # Find timestamp of lowest flexion
            min_t = now
            for t, f in self.flexion_history:
                if f == min_flex_in_window:
                    min_t = t
                    break
            dt_burst = max(now - min_t, 0.001)
            dE_dt = (self.flexion - min_flex_in_window) / dt_burst

            if dE_dt > GestureArbitrator.BLOOM_EXP_VELOCITY:
                self.bloom_triggered = True
                self.flexion_history.clear()

        # ── Biomechanical Snap Tracker with Fist Isolation ───────────────
        # Ring (norm_tip_dists[2]) and Pinky (norm_tip_dists[3]) must be extended (> 1.0 * L_ref)
        ring_ext = norm_tip_dists[2]
        pinky_ext = norm_tip_dists[3]
        snap_fingers_extended = bool(
            ring_ext > GestureArbitrator.SNAP_FINGER_EXT_THRESHOLD and
            pinky_ext > GestureArbitrator.SNAP_FINGER_EXT_THRESHOLD
        )

        self.snap_triggered = False
        snap_dD_dt = (norm_thumb_middle - self.prev_snap_dist) / max(dt, 0.001)
        self.prev_snap_dist = norm_thumb_middle

        if not snap_fingers_extended:
            # Ring or Pinky is curled -> User is forming a fist. Abort snap immediately!
            self.snap_primed = False
            self.snap_prime_counter = 0
            self.c_vision = 0.0
        else:
            # Priming: Thumb & Middle finger touch (< 0.28) for >= 2 frames
            if norm_thumb_middle < GestureArbitrator.SNAP_PRIMED_MAX_DIST:
                self.snap_prime_counter += 1
                if self.snap_prime_counter >= 2:
                    if not self.snap_primed:
                        self.snap_primed = True
                        self.snap_primed_time = now
                        self.snap_primed_middle_y = middle_tip_y
            else:
                if self.snap_primed:
                    prime_dur = now - self.snap_primed_time
                    delta_y = middle_tip_y - self.snap_primed_middle_y # Downward slip

                    if 0.025 <= prime_dur <= 0.220 and (delta_y > 0.06 or snap_dD_dt > 2.4):
                        vel_factor = min(1.0, max(0.0, (snap_dD_dt - 2.4) / 1.6))
                        y_factor = min(1.0, max(0.0, (delta_y - 0.06) / 0.08))
                        self.c_vision = min(1.0, 0.62 + y_factor * 0.20 + vel_factor * 0.18)

                        if self.c_vision >= GestureArbitrator.SNAP_VISION_CONF_THRESHOLD:
                            self.snap_triggered = True

                    self.snap_primed = False
                    self.snap_prime_counter = 0

        self.c_vision = max(0.0, self.c_vision - dt * 2.5)

        self.label = str(label)
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
        self.live_feed_clients = set()

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
        self.hand_states = {}
        self.persistence_counters = {}

        # Dual-hand 1€ filters
        self.dual_dist_filter = OneEuroFilter(min_cutoff=1.2, beta=0.006)
        self.dual_cx_filter   = OneEuroFilter(min_cutoff=1.5, beta=0.008)
        self.dual_cy_filter   = OneEuroFilter(min_cutoff=1.5, beta=0.008)
        self.dual_angle_filter = OneEuroFilter(min_cutoff=1.0, beta=0.005)

        # 60 FPS MJPEG Stream Buffer
        self.latest_jpeg = None
        self.stream_viewers = 0
        self._jpeg_lock = threading.Lock()
        self._async_loop = None

        # Dedicated Ingestion Thread Buffer (maxlen=1)
        self._frame_buffer = collections.deque(maxlen=1)

        # Global Snap Cooldown Refractory Timer (2.0s)
        self.last_snap_time = 0.0

        self.last_frame_time = time.time()
        self.latest_state = {
            "hands": [],
            "state": "IDLE",
            "flexion": 1.4,
            "two_hand_dist": 0.0,
            "dual_angle": 0.0,
            "dual_pinch": False,
            "dual_pinch_center": {"x": 0.5, "y": 0.5},
            "grab_hand": None,
            "bloom": False,
            "compress": False,
            "snap": False,
            "c_vision": 0.0,
            "event": None,
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

    async def register_live_feed(self, ws):
        self.live_feed_clients.add(ws)
        try:
            while True:
                await ws.receive_text()
        except Exception:
            pass
        finally:
            self.live_feed_clients.discard(ws)

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

    async def broadcast_live_feed(self, jpeg_bytes):
        if self.live_feed_clients:
            dead = set()
            for ws in list(self.live_feed_clients):
                try:
                    await ws.send_bytes(jpeg_bytes)
                except Exception:
                    dead.add(ws)
            self.live_feed_clients -= dead

    async def generate_mjpeg_stream(self):
        """Asynchronous non-blocking MJPEG generator."""
        self.stream_viewers += 1
        last_sent = None
        try:
            while self.running:
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
                await asyncio.sleep(0.016)
        except (asyncio.CancelledError, GeneratorExit, Exception):
            pass
        finally:
            self.stream_viewers = max(0, self.stream_viewers - 1)

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
        global_snap = False
        max_c_vision = 0.0
        min_flexion = 1.4

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

                # ── Render 21-Node Skeletal Mesh on Frame ────────────────
                pts_2d = (pts[:, :2] * np.array([w, h], dtype=np.float32)).astype(np.int32)
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts_2d[start_idx]), tuple(pts_2d[end_idx]), (255, 255, 0), 2, cv2.LINE_AA)

                for pt in pts_2d:
                    cv2.circle(frame, tuple(pt), 4, (0, 255, 255), -1, cv2.LINE_AA)

                wrist_pt = (pts_2d[0][0], pts_2d[0][1] + 20)
                cv2.putText(frame, label.upper(), wrist_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # ── Geometric Scale-Invariant Heuristics (L_ref metric) ──
                wrist = pts[0]
                middle_mcp = pts[9]
                index_mcp = pts[5]
                pinky_mcp = pts[17]

                raw_cx = float((wrist[0] + middle_mcp[0]) / 2.0)
                raw_cy = float((wrist[1] + middle_mcp[1]) / 2.0)

                l_ref = float(np.linalg.norm(middle_mcp - wrist))
                safe_l_ref = max(l_ref, 1e-6)
                raw_depth = safe_l_ref * 3.5

                # Pinch Metric: ||Thumb_Tip (4) - Index_Tip (8)|| / L_ref
                thumb_tip = pts[4]
                index_tip = pts[8]
                tip_dist = float(np.linalg.norm(thumb_tip - index_tip))
                pinch_ratio = tip_dist / safe_l_ref
                pinch_center = (float((thumb_tip[0] + index_tip[0]) / 2.0),
                                float((thumb_tip[1] + index_tip[1]) / 2.0))

                # Snap Metric: ||Thumb_Tip (4) - Middle_Tip (12)|| / L_ref
                middle_tip = pts[12]
                thumb_middle_dist = float(np.linalg.norm(thumb_tip - middle_tip))
                norm_thumb_middle = thumb_middle_dist / safe_l_ref

                # 4 Fingertips: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
                four_tips = pts[[8, 12, 16, 20]]
                dists_to_wrist = np.linalg.norm(four_tips - wrist, axis=1)
                norm_tip_dists = dists_to_wrist / safe_l_ref
                raw_flexion = float(np.mean(norm_tip_dists))

                # 3D Palm Euler Vectorization
                v1 = middle_mcp - wrist
                v2 = pinky_mcp - index_mcp
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                normal = normal / norm_len if norm_len > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

                pitch = float(math.asin(max(-1.0, min(1.0, -normal[1]))))
                yaw   = float(math.atan2(normal[0], normal[2]))
                roll  = float(math.atan2(v2[1], math.sqrt(v2[0]**2 + v2[2]**2)))

                # ── Handedness-Keyed Update with Continuous Flexion & Snap ──
                if label not in self.hand_states:
                    self.hand_states[label] = HandState(
                        raw_cx, raw_cy, raw_depth, pinch_ratio, pinch_center,
                        label, raw_flexion, now
                    )
                    self.hand_states[label].pitch = pitch
                    self.hand_states[label].yaw = yaw
                    self.hand_states[label].roll = roll
                else:
                    self.hand_states[label].update(
                        raw_cx, raw_cy, raw_depth, pinch_ratio, pinch_center,
                        label, norm_tip_dists, norm_thumb_middle, float(middle_tip[1]), dt, now,
                        pitch=pitch, yaw=yaw, roll=roll
                    )

                sm = self.hand_states[label]
                speed = math.sqrt(sm.vx**2 + sm.vy**2)

                if speed > GestureArbitrator.SWIPE_VELOCITY_THRESHOLD and sm.is_open:
                    slap_active = True
                    slap_vx = sm.vx
                    slap_vy = sm.vy

                if sm.bloom_triggered:
                    global_bloom = True

                if sm.flexion < min_flexion:
                    min_flexion = sm.flexion

                if sm.c_vision > max_c_vision:
                    max_c_vision = sm.c_vision

                # Pure High-Confidence Isolated Snap (>= 0.62 with 2.0s cooldown)
                if sm.snap_triggered and (now - self.last_snap_time) > GestureArbitrator.SNAP_COOLDOWN_SEC:
                    global_snap = True
                    self.last_snap_time = now

        # ── Persistence Cleanup ──────────────────────────────────────────
        all_tracked = list(self.hand_states.keys())
        for label in all_tracked:
            if label not in detected_labels:
                cnt = self.persistence_counters.get(label, 0) - 1
                self.persistence_counters[label] = cnt
                if cnt <= 0:
                    del self.hand_states[label]
                    self.persistence_counters.pop(label, None)

        # ── Hands Telemetry Payload Construction ─────────────────────────
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
                "flexion": round(sm.flexion, 3),
                "is_pinching": sm.is_pinching,
                "is_open": sm.is_open,
                "is_fist": sm.is_fist,
                "c_vision": round(sm.c_vision, 3),
                "pitch": round(sm.pitch, 4),
                "yaw": round(sm.yaw, 4),
                "roll": round(sm.roll, 4),
            })

        # ── Multi-Hand & Pinch Mode Arbitration ──────────────────────────
        two_hand_dist = 0.0
        dual_angle = 0.0
        dual_pinch = False
        dual_pinch_center = {"x": 0.5, "y": 0.5}
        grab_hand = None

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
        elif len(pinching_hands) == 1:
            grab_hand = pinching_hands[0]["id"]
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

        # Determine Dominant State
        state = GestureArbitrator.STATE_IDLE
        event = None
        if global_snap:
            state = GestureArbitrator.STATE_SNAP
            event = "SNAP"
        elif global_bloom:
            state = GestureArbitrator.STATE_BLOOM
        elif dual_pinch:
            state = GestureArbitrator.STATE_DUAL_PINCH
        elif grab_hand is not None:
            state = GestureArbitrator.STATE_GRAB
        elif min_flexion <= 0.65:
            state = GestureArbitrator.STATE_COMPRESS
        elif slap_active:
            state = GestureArbitrator.STATE_SWIPE
        elif len(hands_data) > 0:
            state = GestureArbitrator.STATE_HOVER

        self.latest_state = {
            "hands": hands_data,
            "state": state,
            "event": event,
            "flexion": round(min_flexion, 3),
            "two_hand_dist": two_hand_dist,
            "dual_angle": round(dual_angle, 4),
            "dual_pinch": dual_pinch,
            "dual_pinch_center": dual_pinch_center,
            "grab_hand": grab_hand,
            "bloom": global_bloom,
            "compress": bool(min_flexion <= 0.65),
            "snap": global_snap,
            "c_vision": round(max_c_vision, 3),
            "slap_impulse": {"active": slap_active, "vx": slap_vx, "vy": slap_vy}
        }

        # ── True 60 FPS Binary Frame Push (480x270 @ Quality 60) ────────
        if len(self.live_feed_clients) > 0 or self.stream_viewers > 0:
            preview = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_LINEAR)
            ret, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                jpeg_bytes = jpeg.tobytes()
                with self._jpeg_lock:
                    self.latest_jpeg = jpeg_bytes
                if self.live_feed_clients and self._async_loop and not self._async_loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self.broadcast_live_feed(jpeg_bytes), self._async_loop)

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

        # Background Ingestion Worker Thread
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

        # Processing Loop
        while self.running:
            if self._frame_buffer:
                raw_frame = self._frame_buffer.pop()
                flipped = cv2.flip(raw_frame, 1)
                self.process_frame(flipped)
                asyncio.run_coroutine_threadsafe(self.broadcast(), loop)
            else:
                time.sleep(0.002)

        cap.release()
