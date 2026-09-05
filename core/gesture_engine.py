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


def compute_vector_curl(pts):
    """
    Occlusion-Proof Fist Detection:
    Replace distance-to-wrist formulas with 3D joint angle and palm normal vector checks.
    A finger is curled if the Y-coordinate of its Tip (Nodes 8, 12, 16, 20) is closer
    to the Wrist (Node 0) than its corresponding PIP joint (Nodes 6, 10, 14, 18) when
    projected along the palm normal vector into the longitudinal palm coordinate frame.
    Continuously maps this average vector curl to E_avg (1.0 = open, 0.0 = tight fist).
    """
    wrist = pts[0]
    middle_mcp = pts[9]
    index_mcp = pts[5]
    pinky_mcp = pts[17]

    v1 = middle_mcp - wrist
    v2 = pinky_mcp - index_mcp
    normal = np.cross(v1, v2)
    n_len = np.linalg.norm(normal)
    normal = normal / n_len if n_len > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    l_ref = max(float(np.linalg.norm(v1)), 1e-6)
    u_y = v1 / l_ref  # longitudinal unit vector pointing toward fingers

    finger_pairs = [
        (8, 6),   # Index Tip & PIP
        (12, 10), # Middle Tip & PIP
        (16, 14), # Ring Tip & PIP
        (20, 18), # Pinky Tip & PIP
    ]

    curls = []
    curled_flags = []

    for tip_idx, pip_idx in finger_pairs:
        tip = pts[tip_idx]
        pip = pts[pip_idx]

        # Project along palm normal into palm plane, then measure along longitudinal Y axis
        # y = dot(P - wrist, u_y)
        y_tip = float(np.dot(tip - wrist, u_y))
        y_pip = float(np.dot(pip - wrist, u_y))

        # A finger is curled if Tip is closer to Wrist than PIP
        is_curled = bool(y_tip < y_pip)
        curled_flags.append(is_curled)

        # Continuous extension metric:
        # Fully open: diff ~ 0.5 to 0.7 L_ref -> E ~ 1.0
        # Fully tight fist: diff ~ -0.2 to -0.4 L_ref -> E ~ 0.0
        diff = (y_tip - y_pip) / l_ref
        e = float(np.clip((diff + 0.30) / 0.90, 0.0, 1.0))
        curls.append(e)

    e_avg = float(np.mean(curls))
    all_curled = all(curled_flags)
    is_fist = bool(all_curled or e_avg <= 0.25)
    is_open = bool(all(not c for c in curled_flags) and e_avg >= 0.70)

    return e_avg, is_fist, is_open, curled_flags


# ---------------------------------------------------------------------------
# Per-Hand Tracked State with Kinematic Vector Tracking
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
        self.is_open = bool(raw_flexion >= 0.70)
        self.is_fist = bool(raw_flexion <= 0.25)

        # Continuous Flexion Tracking & Bloom
        self.flexion_history = collections.deque(maxlen=20)
        self.flexion_history.append((now, self.flexion))
        self.bloom_triggered = False

        # Fist Hysteresis State for Bloom Gating
        self.fist_entered_time = 0.0
        self.fist_latched = False
        self._open_frame_counter = 0

        # 3D Palm Normal Euler angles
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

    def update(self, raw_cx, raw_cy, raw_depth, raw_pinch, pinch_pos, label,
               e_avg, is_curled_fist, pts, safe_l_ref, dt, now,
               pitch=0.0, yaw=0.0, roll=0.0):
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

        # Continuous Knuckle Extension Telemetry (1€ Filtered E_avg: 1.0 = open, 0.0 = tight fist)
        self.flexion = float(self.filters["flexion"](e_avg, now))
        self.flexion_history.append((now, self.flexion))

        while len(self.flexion_history) > 1 and (now - self.flexion_history[0][0]) > 0.400:
            self.flexion_history.popleft()

        self.is_open = bool(self.flexion >= 0.70)
        self.is_fist = bool(is_curled_fist or self.flexion <= 0.25)

        # Strict Fist Hysteresis for Bloom Gating
        self.bloom_triggered = False

        if self.is_fist:
            if not self.fist_latched:
                self.fist_latched = True
                self.fist_entered_time = now
            self._open_frame_counter = 0
        else:
            self._open_frame_counter += 1
            if self.fist_latched:
                if self._open_frame_counter >= 3:
                    fist_hold_duration = now - self.fist_entered_time
                    if fist_hold_duration >= GestureArbitrator.BLOOM_FIST_HOLD_MS and self.is_open:
                        self.bloom_triggered = True
                        self.fist_latched = False
                        self.fist_entered_time = 0.0
                    elif self._open_frame_counter > 8:
                        self.fist_latched = False

        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        if self.pinch < GestureArbitrator.PINCH_ENTER_RATIO:
            self.is_pinching = True
        elif self.pinch > GestureArbitrator.PINCH_EXIT_RATIO:
            self.is_pinching = False


class GestureEngine:
    min_detection_confidence = 0.75
    min_tracking_confidence = 0.85
    min_hand_detection_confidence = 0.75
    min_hand_presence_confidence = 0.85

    def __init__(self):
        ensure_model()
        self.connected_clients = set()
        self.live_feed_clients = set()

        # MediaPipe Hardening: min_detection_confidence=0.75, min_tracking_confidence=0.85
        self.min_detection_confidence = 0.75
        self.min_tracking_confidence = 0.85
        self.min_hand_detection_confidence = 0.75
        self.min_hand_presence_confidence = 0.85

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.75,
            min_hand_presence_confidence=0.85,
            min_tracking_confidence=0.85
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.running = True

        # Pre-Kinematic Rolling EMA Smoothing Dictionary for all 21 joint coordinates
        self.ema_landmarks = {}

        # Anti-Glitch Coasting Buffer (5 frames temporal persistence)
        self.coasting_frames = {}
        self.last_valid_pts = {}

        # Handedness-keyed tracking dictionaries
        self.hand_states = {}

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
        self.last_frame_time = time.time()
        self.latest_state = {
            "hands": [],
            "state": "IDLE",
            "flexion": 1.0,
            "two_hand_dist": 0.0,
            "dual_angle": 0.0,
            "dual_pinch": False,
            "dual_pinch_center": {"x": 0.5, "y": 0.5},
            "grab_hand": None,
            "bloom": False,
            "compress": False,
            "slap_impulse": {"active": False, "vx": 0.0, "vy": 0.0}
        }

    def register_client(self, ws):
        self.connected_clients.add(ws)

    def unregister_client(self, ws):
        self.connected_clients.discard(ws)

    def register_live_feed_client(self, ws):
        self.live_feed_clients.add(ws)

    def unregister_live_feed_client(self, ws):
        self.live_feed_clients.discard(ws)

    async def broadcast(self):
        if not self.connected_clients:
            return
        payload = json.dumps(self.latest_state)
        dead = []
        for ws in self.connected_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connected_clients.discard(ws)

    async def broadcast_live_feed(self, jpeg_bytes: bytes):
        if not self.live_feed_clients:
            return
        dead = []
        for ws in self.live_feed_clients:
            try:
                await ws.send_bytes(jpeg_bytes)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.live_feed_clients.discard(ws)

    async def generate_mjpeg(self):
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
        min_flexion = 1.0

        # Active hands dictionary for this frame (label -> smoothed_pts)
        active_landmarks_dict = {}

        # 1. Inspect MediaPipe detections with confidence check and Pre-Kinematic EMA
        if res.hand_landmarks:
            for idx, lms in enumerate(res.hand_landmarks):
                label = "Right"
                score = 1.0
                if res.handedness and idx < len(res.handedness) and len(res.handedness[idx]) > 0:
                    label = res.handedness[idx][0].category_name
                    score = float(res.handedness[idx][0].score)

                if label in active_landmarks_dict:
                    label = f"{label}_2"

                raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                # MediaPipe Hardening Check (0.85 confidence threshold)
                if score >= 0.85:
                    # Pre-Kinematic EMA Smoothing: smoothed = (raw * 0.4) + (previous * 0.6)
                    if label in self.ema_landmarks:
                        smoothed = (raw_pts * 0.4) + (self.ema_landmarks[label] * 0.6)
                    else:
                        smoothed = raw_pts.copy()

                    self.ema_landmarks[label] = smoothed
                    self.last_valid_pts[label] = smoothed.copy()
                    self.coasting_frames[label] = 0
                    active_landmarks_dict[label] = smoothed
                else:
                    # Tracker confidence dropped below 0.85: coast using previous frame's smoothed vectors
                    coast_cnt = self.coasting_frames.get(label, 0) + 1
                    self.coasting_frames[label] = coast_cnt
                    if coast_cnt <= 5 and label in self.last_valid_pts:
                        active_landmarks_dict[label] = self.last_valid_pts[label]

        # 2. Anti-Glitch Coasting: check previously tracked hands missed in this frame
        for prev_label in list(self.last_valid_pts.keys()):
            if prev_label not in active_landmarks_dict:
                coast_cnt = self.coasting_frames.get(prev_label, 0) + 1
                self.coasting_frames[prev_label] = coast_cnt
                if coast_cnt <= 5:
                    # Coast using previous frame's smoothed vectors (up to 5 frames)
                    active_landmarks_dict[prev_label] = self.last_valid_pts[prev_label]
                else:
                    # Exceeded 5 frames: disconnect hand
                    self.coasting_frames.pop(prev_label, None)
                    self.last_valid_pts.pop(prev_label, None)
                    self.ema_landmarks.pop(prev_label, None)
                    self.hand_states.pop(prev_label, None)

        # 3. Process kinematics for all active (detected + coasting) hands
        if active_landmarks_dict:
            for label, pts in active_landmarks_dict.items():
                # Render 21-Node Skeletal Mesh on Frame
                pts_2d = (pts[:, :2] * np.array([w, h], dtype=np.float32)).astype(np.int32)
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts_2d[start_idx]), tuple(pts_2d[end_idx]), (255, 255, 0), 2, cv2.LINE_AA)

                for pt in pts_2d:
                    cv2.circle(frame, tuple(pt), 4, (0, 255, 255), -1, cv2.LINE_AA)

                wrist_pt = (pts_2d[0][0], pts_2d[0][1] + 20)
                cv2.putText(frame, label.upper(), wrist_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # Geometric Scale-Invariant Landmarks
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

                # Occlusion-Proof Fist & Extension via Palm Normal Projection
                e_avg, is_curled_fist, is_open_hand, curled_flags = compute_vector_curl(pts)

                # 3D Palm Euler Angles
                v1 = middle_mcp - wrist
                v2 = pinky_mcp - index_mcp
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                normal = normal / norm_len if norm_len > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

                pitch = float(math.asin(max(-1.0, min(1.0, -normal[1]))))
                yaw   = float(math.atan2(normal[0], normal[2]))
                roll  = float(math.atan2(v2[1], math.sqrt(v2[0]**2 + v2[2]**2)))

                # Update or Instantiate HandState
                if label not in self.hand_states:
                    self.hand_states[label] = HandState(
                        raw_cx, raw_cy, raw_depth, pinch_ratio, pinch_center,
                        label, e_avg, now
                    )
                    self.hand_states[label].pitch = pitch
                    self.hand_states[label].yaw = yaw
                    self.hand_states[label].roll = roll
                else:
                    self.hand_states[label].update(
                        raw_cx, raw_cy, raw_depth, pinch_ratio, pinch_center,
                        label, e_avg, is_curled_fist, pts, safe_l_ref, dt, now,
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

        # Clean up any hand states not in active_landmarks_dict
        for label in list(self.hand_states.keys()):
            if label not in active_landmarks_dict:
                del self.hand_states[label]

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
        if global_bloom:
            state = GestureArbitrator.STATE_BLOOM
        elif dual_pinch:
            state = GestureArbitrator.STATE_DUAL_PINCH
        elif grab_hand is not None:
            state = GestureArbitrator.STATE_GRAB
        elif min_flexion <= 0.35:
            state = GestureArbitrator.STATE_COMPRESS
        elif slap_active:
            state = GestureArbitrator.STATE_SWIPE
        elif len(hands_data) > 0:
            state = GestureArbitrator.STATE_HOVER

        self.latest_state = {
            "hands": hands_data,
            "state": state,
            "flexion": round(min_flexion, 3),
            "two_hand_dist": two_hand_dist,
            "dual_angle": round(dual_angle, 4),
            "dual_pinch": dual_pinch,
            "dual_pinch_center": dual_pinch_center,
            "grab_hand": grab_hand,
            "bloom": global_bloom,
            "compress": bool(min_flexion <= 0.35),
            "slap_impulse": {"active": slap_active, "vx": slap_vx, "vy": slap_vy}
        }

        # ── True 60 FPS Binary Frame Push (640x360 @ Quality 85) ────────
        if len(self.live_feed_clients) > 0 or self.stream_viewers > 0:
            preview = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
            ret, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
