import asyncio
import json
import math
import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

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
# One Euro (1€) Adaptive Filter
# ---------------------------------------------------------------------------
class OneEuroFilter:
    """Adaptive frequency-domain filter for jitter-free, zero-lag tracking."""
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _smoothing_factor(te, cutoff):
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        te = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# ---------------------------------------------------------------------------
# Per-hand tracked state keyed by handedness ("Left" / "Right")
# ---------------------------------------------------------------------------
class HandState:
    def __init__(self, raw_cx, raw_cy, raw_depth, raw_pinch, label, is_open, avg_angle, now):
        self.filters = {
            "x":     OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "y":     OneEuroFilter(min_cutoff=1.5, beta=0.008),
            "depth": OneEuroFilter(min_cutoff=1.0, beta=0.005),
            "pinch": OneEuroFilter(min_cutoff=2.0, beta=0.004),
        }
        self.x = self.filters["x"](raw_cx, now)
        self.y = self.filters["y"](raw_cy, now)
        self.depth = self.filters["depth"](raw_depth, now)
        self.pinch = self.filters["pinch"](raw_pinch, now)

        self.vx = 0.0
        self.vy = 0.0
        self.prev_x = raw_cx
        self.prev_y = raw_cy
        self.is_pinching = raw_pinch < 0.22
        self.label = label
        self.is_open = is_open

        # Knuckle flexion & bloom tracking
        self.avg_angle = avg_angle
        self.prev_avg_angle = avg_angle
        self.angle_vel = 0.0
        self.is_fist = avg_angle < 65.0
        self.bloom_triggered = False

        # 3D palm normal Euler angles
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

    def update(self, raw_cx, raw_cy, raw_depth, raw_pinch, label, is_open, avg_angle, dt, now,
               pitch=0.0, yaw=0.0, roll=0.0):
        self.x = self.filters["x"](raw_cx, now)
        self.y = self.filters["y"](raw_cy, now)
        self.depth = self.filters["depth"](raw_depth, now)
        self.pinch = self.filters["pinch"](raw_pinch, now)

        vx = (self.x - self.prev_x) / dt
        vy = (self.y - self.prev_y) / dt
        self.vx = self.vx * 0.55 + vx * 0.45
        self.vy = self.vy * 0.55 + vy * 0.45
        self.prev_x = self.x
        self.prev_y = self.y

        d_angle_deg = (avg_angle - self.prev_avg_angle) / dt
        d_angle_rad = d_angle_deg * (math.pi / 180.0)
        self.angle_vel = self.angle_vel * 0.4 + d_angle_rad * 0.6
        self.prev_avg_angle = avg_angle
        self.avg_angle = avg_angle

        was_fist = self.is_fist
        self.is_fist = avg_angle < 65.0
        self.bloom_triggered = (was_fist or avg_angle < 85.0) and (self.angle_vel > 2.8) and (avg_angle > 110.0)

        self.label = label
        self.is_open = is_open
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        # Scale-invariant pinch hysteresis
        if self.is_pinching:
            if self.pinch > 0.32:
                self.is_pinching = False
        else:
            if self.pinch < 0.22:
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

        # Handedness-keyed tracking dictionaries (prevents 0 vs 1 index swapping)
        self.hand_states = {}           # label -> HandState
        self.persistence_counters = {}  # label -> int

        # Dual-hand 1€ filters for rock-solid stability
        self.dual_dist_filter = OneEuroFilter(min_cutoff=1.2, beta=0.006)
        self.dual_cx_filter   = OneEuroFilter(min_cutoff=1.5, beta=0.008)
        self.dual_cy_filter   = OneEuroFilter(min_cutoff=1.5, beta=0.008)
        self.dual_angle_filter = OneEuroFilter(min_cutoff=1.0, beta=0.005)

        self.latest_jpeg = None
        self.last_frame_time = time.time()
        self.latest_state = {
            "hands": [],
            "pinch_priority_hand": None,
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

    async def generate_mjpeg(self):
        """Asynchronous generator yielding live JPEG frames for in-browser HUD streaming."""
        while self.running:
            if self.latest_jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + self.latest_jpeg + b"\r\n"
                )
            await asyncio.sleep(0.033)  # ~30 fps browser preview

    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    @staticmethod
    def _vec3(p):
        return (p.x, p.y, p.z)

    @staticmethod
    def _sub(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

    @staticmethod
    def _cross(a, b):
        return (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        )

    @staticmethod
    def _norm(v):
        m = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        return (v[0]/m, v[1]/m, v[2]/m) if m > 1e-9 else (0, 0, 0)

    def _joint_angle(self, a, b, c):
        v1 = self._sub(self._vec3(a), self._vec3(b))
        v2 = self._sub(self._vec3(c), self._vec3(b))
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        m1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        m2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        if m1 * m2 < 1e-9:
            return 180.0
        cos_ang = max(-1.0, min(1.0, dot / (m1 * m2)))
        return math.degrees(math.acos(cos_ang))

    def _palm_euler(self, lms):
        wrist = self._vec3(lms[0])
        middle_mcp = self._vec3(lms[9])
        index_mcp = self._vec3(lms[5])
        pinky_mcp = self._vec3(lms[17])

        v1 = self._sub(middle_mcp, wrist)
        v2 = self._sub(pinky_mcp, index_mcp)
        normal = self._norm(self._cross(v1, v2))

        nx, ny, nz = normal
        pitch = math.asin(max(-1.0, min(1.0, -ny)))
        yaw   = math.atan2(nx, nz)
        roll  = math.atan2(v2[1], math.sqrt(v2[0]**2 + v2[2]**2))
        return pitch, yaw, roll

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
                # ── Handedness resolution ────────────────────────────────
                label = "Right"
                if res.handedness and idx < len(res.handedness):
                    label = res.handedness[idx][0].category_name

                # Avoid collision if MediaPipe momentarily labels two hands with same name
                if label in detected_labels:
                    label = f"{label}_2"

                detected_labels.add(label)
                self.persistence_counters[label] = 4  # 4-frame persistence

                # ── Draw 21-node skeletal mesh on frame for in-browser stream ──
                for start_idx, end_idx in HAND_CONNECTIONS:
                    pt1 = (int(lms[start_idx].x * w), int(lms[start_idx].y * h))
                    pt2 = (int(lms[end_idx].x * w), int(lms[end_idx].y * h))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

                for lm in lms:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

                # Hand label indicator
                wrist_pt = (int(lms[0].x * w), int(lms[0].y * h) + 20)
                cv2.putText(frame, label.upper(), wrist_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # ── Kinematic metrics ────────────────────────────────────
                raw_cx = (lms[0].x + lms[9].x) / 2.0
                raw_cy = (lms[0].y + lms[9].y) / 2.0
                knuckle_len = self.dist(lms[9], lms[0])
                raw_depth = knuckle_len * 3.5

                tip_dist = self.dist(lms[4], lms[8])
                pinch_ratio = tip_dist / knuckle_len if knuckle_len > 1e-6 else 1.0

                ang_index  = self._joint_angle(lms[5], lms[6], lms[8])
                ang_middle = self._joint_angle(lms[9], lms[10], lms[12])
                ang_ring   = self._joint_angle(lms[13], lms[14], lms[16])
                ang_pinky  = self._joint_angle(lms[17], lms[18], lms[20])
                avg_angle = (ang_index + ang_middle + ang_ring + ang_pinky) / 4.0

                is_open = avg_angle > 140.0
                pitch, yaw, roll = self._palm_euler(lms)

                # ── Handedness-keyed 1€ update ───────────────────────────
                if label not in self.hand_states:
                    self.hand_states[label] = HandState(
                        raw_cx, raw_cy, raw_depth, pinch_ratio,
                        label, is_open, avg_angle, now
                    )
                    self.hand_states[label].pitch = pitch
                    self.hand_states[label].yaw = yaw
                    self.hand_states[label].roll = roll
                else:
                    self.hand_states[label].update(
                        raw_cx, raw_cy, raw_depth, pinch_ratio,
                        label, is_open, avg_angle, dt, now,
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

        # ── 4-frame persistence cleanup ──────────────────────────────────
        all_tracked = list(self.hand_states.keys())
        for label in all_tracked:
            if label not in detected_labels:
                cnt = self.persistence_counters.get(label, 0) - 1
                self.persistence_counters[label] = cnt
                if cnt <= 0:
                    del self.hand_states[label]
                    self.persistence_counters.pop(label, None)

        # ── Multi-Hand Priority Arbitration ──────────────────────────────
        pinching_hands = [sm for sm in self.hand_states.values() if sm.is_pinching]
        pinch_priority_hand = pinching_hands[0].label if len(pinching_hands) == 1 else None

        for label, sm in self.hand_states.items():
            hands_data.append({
                "id": label,
                "label": sm.label,
                "palm_x": sm.x,
                "palm_y": sm.y,
                "depth_scale": min(max(sm.depth, 0.5), 2.2),
                "pinch_dist": sm.pinch,
                "is_pinching": sm.is_pinching,
                "is_open": sm.is_open,
                "is_fist": sm.is_fist,
                "avg_angle": round(sm.avg_angle, 2),
                "pitch": round(sm.pitch, 4),
                "yaw": round(sm.yaw, 4),
                "roll": round(sm.roll, 4),
            })

        two_hand_dist = 0.0
        dual_angle = 0.0
        dual_pinch = False
        dual_pinch_center = {"x": 0.5, "y": 0.5}

        if len(hands_data) == 2:
            p1, p2 = hands_data[0], hands_data[1]
            raw_dx = p2["palm_x"] - p1["palm_x"]
            raw_dy = p2["palm_y"] - p1["palm_y"]
            raw_dist = math.sqrt(raw_dx**2 + raw_dy**2)
            raw_angle = math.atan2(raw_dy, raw_dx)
            raw_cx = (p1["palm_x"] + p2["palm_x"]) / 2.0
            raw_cy = (p1["palm_y"] + p2["palm_y"]) / 2.0

            # 1€ filtered dual metrics for rock-solid stability
            two_hand_dist = self.dual_dist_filter(raw_dist, now)
            dual_angle = self.dual_angle_filter(raw_angle, now)
            dual_pinch_center = {
                "x": self.dual_cx_filter(raw_cx, now),
                "y": self.dual_cy_filter(raw_cy, now)
            }

            if p1["is_pinching"] and p2["is_pinching"]:
                dual_pinch = True

        self.latest_state = {
            "hands": hands_data,
            "pinch_priority_hand": pinch_priority_hand,
            "two_hand_dist": two_hand_dist,
            "dual_angle": round(dual_angle, 4),
            "dual_pinch": dual_pinch,
            "dual_pinch_center": dual_pinch_center,
            "bloom": global_bloom,
            "compress": global_compress,
            "slap_impulse": {"active": slap_active, "vx": slap_vx, "vy": slap_vy}
        }

        # Encode live JPEG for in-browser streaming endpoint
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ret:
            self.latest_jpeg = jpeg.tobytes()

        return frame

    def run_capture(self, loop):
        cap = cv2.VideoCapture(0)

        # 60 FPS hardware capture optimization
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Headless capture loop (NO cv2.imshow / waitKey desktop pop-ups)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            self.process_frame(frame)

            asyncio.run_coroutine_threadsafe(self.broadcast(), loop)

        cap.release()
