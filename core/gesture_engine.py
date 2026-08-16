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
        self.smoothed_hands = {}
        self.persistence_counters = {}
        self.last_frame_time = time.time()
        self.latest_state = {
            "hands": [],
            "two_hand_dist": 0.0,
            "dual_pinch": False,
            "dual_pinch_center": {"x": 0.5, "y": 0.5},
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

    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def process_frame(self, frame):
        now = time.time()
        dt = max(now - self.last_frame_time, 0.001)
        self.last_frame_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)

        h, w, _ = frame.shape
        hands_data = []
        alpha = 0.25
        slap_active = False
        slap_vx, slap_vy = 0.0, 0.0

        detected_indices = set()

        if res.hand_landmarks:
            for idx, lms in enumerate(res.hand_landmarks):
                detected_indices.add(idx)
                self.persistence_counters[idx] = 4  # Reset 4-frame persistence

                # Draw complete 21-point skeletal mesh on video frame
                # 1. Bone connections in Cyan (BGR: 255, 255, 0)
                for start_idx, end_idx in HAND_CONNECTIONS:
                    pt1 = (int(lms[start_idx].x * w), int(lms[start_idx].y * h))
                    pt2 = (int(lms[end_idx].x * w), int(lms[end_idx].y * h))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

                # 2. Joint landmarks in Yellow (BGR: 0, 255, 255)
                for lm in lms:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

                label = "Right"
                if res.handedness and idx < len(res.handedness):
                    label = res.handedness[idx][0].category_name

                raw_cx = (lms[0].x + lms[9].x) / 2.0
                raw_cy = (lms[0].y + lms[9].y) / 2.0
                raw_depth = self.dist(lms[0], lms[9]) * 3.5
                raw_pinch = self.dist(lms[4], lms[8])

                is_open = sum([
                    self.dist(lms[8], lms[0]) > self.dist(lms[6], lms[0]),
                    self.dist(lms[12], lms[0]) > self.dist(lms[10], lms[0]),
                    self.dist(lms[16], lms[0]) > self.dist(lms[14], lms[0]),
                    self.dist(lms[20], lms[0]) > self.dist(lms[18], lms[0])
                ]) >= 4

                if idx not in self.smoothed_hands:
                    self.smoothed_hands[idx] = {
                        "x": raw_cx, "y": raw_cy, "depth": raw_depth, "pinch": raw_pinch,
                        "vx": 0.0, "vy": 0.0, "is_pinching": raw_pinch < 0.055, "label": label,
                        "is_open": is_open
                    }
                else:
                    prev = self.smoothed_hands[idx]
                    vx = (raw_cx - prev["x"]) / dt
                    vy = (raw_cy - prev["y"]) / dt
                    prev["vx"] = prev["vx"] * 0.6 + vx * 0.4
                    prev["vy"] = prev["vy"] * 0.6 + vy * 0.4
                    prev["x"] += alpha * (raw_cx - prev["x"])
                    prev["y"] += alpha * (raw_cy - prev["y"])
                    prev["depth"] += alpha * (raw_depth - prev["depth"])
                    prev["pinch"] += alpha * (raw_pinch - prev["pinch"])
                    prev["label"] = label
                    prev["is_open"] = is_open

                    # Pinch hysteresis thresholding
                    if prev["is_pinching"]:
                        if prev["pinch"] > 0.070:
                            prev["is_pinching"] = False
                    else:
                        if prev["pinch"] < 0.050:
                            prev["is_pinching"] = True

                sm = self.smoothed_hands[idx]
                speed = math.sqrt(sm["vx"]**2 + sm["vy"]**2)
                if speed > 1.6 and is_open:
                    slap_active = True
                    slap_vx = sm["vx"]
                    slap_vy = sm["vy"]

        # Handle temporal persistence coasting for momentarily dropped hands
        all_tracked_indices = list(self.smoothed_hands.keys())
        for idx in all_tracked_indices:
            if idx not in detected_indices:
                cnt = self.persistence_counters.get(idx, 0) - 1
                self.persistence_counters[idx] = cnt
                if cnt <= 0:
                    del self.smoothed_hands[idx]
                    if idx in self.persistence_counters:
                        del self.persistence_counters[idx]

        for idx, sm in self.smoothed_hands.items():
            hands_data.append({
                "id": idx,
                "label": sm.get("label", "Hand"),
                "palm_x": sm["x"],
                "palm_y": sm["y"],
                "depth_scale": min(max(sm["depth"], 0.5), 2.2),
                "pinch_dist": sm["pinch"],
                "is_pinching": sm["is_pinching"],
                "is_open": sm.get("is_open", False)
            })

        two_hand_dist = 0.0
        dual_pinch = False
        dual_pinch_center = {"x": 0.5, "y": 0.5}

        if len(hands_data) == 2:
            p1 = hands_data[0]
            p2 = hands_data[1]
            two_hand_dist = math.sqrt((p1["palm_x"] - p2["palm_x"])**2 + (p1["palm_y"] - p2["palm_y"])**2)
            if p1["is_pinching"] and p2["is_pinching"]:
                dual_pinch = True
                dual_pinch_center = {
                    "x": (p1["palm_x"] + p2["palm_x"]) / 2.0,
                    "y": (p1["palm_y"] + p2["palm_y"]) / 2.0
                }

        self.latest_state = {
            "hands": hands_data,
            "two_hand_dist": two_hand_dist,
            "dual_pinch": dual_pinch,
            "dual_pinch_center": dual_pinch_center,
            "slap_impulse": {"active": slap_active, "vx": slap_vx, "vy": slap_vy}
        }
        return frame

    def run_capture(self, loop):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            processed = self.process_frame(frame)
            cv2.imshow("PIPO Vision Sensor", processed)

            asyncio.run_coroutine_threadsafe(self.broadcast(), loop)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
