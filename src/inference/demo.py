#!/usr/bin/env python
"""
Real-time Webcam Demo for ISL Translation (60fps optimized).

Usage:
    python src/inference/demo.py --checkpoint checkpoints/best.pt
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time
import threading
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.isl_model import ISLTranslator

# ── Threaded camera capture ──────────────────────────────────────────────────

class CameraStream:
    """Threaded camera capture for zero-latency frame reads."""

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self._lock = threading.Lock()
        self._stopped = False
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stopped = True
        self.cap.release()


# ── Lightweight keypoint extraction (MediaPipe Holistic) ─────────────────────

class FastKeypointExtractor:
    """
    Optimized MediaPipe Holistic extractor.
    Uses model_complexity=0 (lite) for speed.
    """

    def __init__(self, complexity: int = 0, min_det: float = 0.5, min_track: float = 0.5):
        import mediapipe as mp
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=complexity,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
            smooth_landmarks=True,
        )
        self.NUM_POSE = 33
        self.NUM_FACE = 468
        self.NUM_HAND = 21

    def extract(self, frame_rgb: np.ndarray):
        """Extract 543 keypoints from an RGB frame. Returns (543, 3) array and detection metadata."""
        results = self.holistic.process(frame_rgb)

        total = self.NUM_POSE + self.NUM_FACE + self.NUM_HAND * 2
        keypoints = np.zeros((total, 3), dtype=np.float32)
        meta = {"pose": False, "face": False, "lh": False, "rh": False}
        idx = 0

        if results.pose_landmarks:
            meta["pose"] = True
            for lm in results.pose_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.visibility]
                idx += 1
        else:
            idx += self.NUM_POSE

        if results.face_landmarks:
            meta["face"] = True
            for lm in results.face_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        else:
            idx += self.NUM_FACE

        if results.left_hand_landmarks:
            meta["lh"] = True
            for lm in results.left_hand_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        else:
            idx += self.NUM_HAND

        if results.right_hand_landmarks:
            meta["rh"] = True
            for lm in results.right_hand_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]
                idx += 1

        return keypoints, meta

    def close(self):
        self.holistic.close()


# ── Drawing utilities ────────────────────────────────────────────────────────

_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
]

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def draw_skeleton(frame, keypoints, meta):
    """Draw skeletal overlay with connections for pose and hands."""
    h, w = frame.shape[:2]
    POSE_OFF = 0
    FACE_OFF = 33
    LH_OFF = 33 + 468
    RH_OFF = 33 + 468 + 21

    # ── Pose skeleton (green) ──
    if meta["pose"]:
        pts = {}
        for i in range(33):
            x, y, c = keypoints[POSE_OFF + i]
            if c > 0.3:
                px, py = int(x * w), int(y * h)
                pts[i] = (px, py)
                cv2.circle(frame, (px, py), 3, (0, 220, 120), -1)
        for a, b in _POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2, cv2.LINE_AA)

    # ── Left hand (coral) ──
    if meta["lh"]:
        pts = {}
        for i in range(21):
            x, y, _ = keypoints[LH_OFF + i]
            if x > 0 or y > 0:
                px, py = int(x * w), int(y * h)
                pts[i] = (px, py)
                cv2.circle(frame, (px, py), 4, (80, 80, 255), -1)
        for a, b in _HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (60, 60, 220), 2, cv2.LINE_AA)

    # ── Right hand (cyan) ──
    if meta["rh"]:
        pts = {}
        for i in range(21):
            x, y, _ = keypoints[RH_OFF + i]
            if x > 0 or y > 0:
                px, py = int(x * w), int(y * h)
                pts[i] = (px, py)
                cv2.circle(frame, (px, py), 4, (255, 200, 0), -1)
        for a, b in _HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (220, 180, 0), 2, cv2.LINE_AA)

    return frame


def draw_hud(frame, glosses, fps, buf_len):
    """Draw a translucent heads-up display with FPS and predictions."""
    h, w = frame.shape[:2]

    # Semi-transparent banner at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # FPS counter (top-right)
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(frame, fps_text, (w - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)

    # Buffer bar (top-right, below FPS)
    bar_w = 120
    bar_fill = min(buf_len / 64, 1.0)
    cv2.rectangle(frame, (w - 150, 45), (w - 150 + bar_w, 55), (60, 60, 60), -1)
    cv2.rectangle(frame, (w - 150, 45), (w - 150 + int(bar_w * bar_fill), 55), (0, 200, 180), -1)
    cv2.putText(frame, f"Buf: {buf_len}", (w - 150, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Gloss output (top-left)
    gloss_text = " ".join(glosses) if glosses else "(signing...)"
    cv2.putText(frame, f"Glosses: {gloss_text}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Instructions (bottom)
    cv2.putText(frame, "Press Q to quit | C to clear buffer", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


# ── Main demo loop ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ISL Translator — Real-time Demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Load model ──
    import yaml
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", {})
    trans_cfg = model_cfg.get("transformer", {})
    stgcn_cfg = model_cfg.get("stgcn", {})

    print("[1/3] Loading model weights...")
    model = ISLTranslator.load(
        args.checkpoint,
        use_translation=False,
        d_model=trans_cfg.get("d_model", 256),
        nhead=trans_cfg.get("nhead", 8),
        num_encoder_layers=trans_cfg.get("num_encoder_layers", 4),
        stgcn_hidden=stgcn_cfg.get("hidden_channels", 64),
        stgcn_layers=stgcn_cfg.get("num_layers", 4),
    )
    model = model.to(device).eval()

    # ── Init camera and extractor ──
    print("[2/3] Starting camera stream...")
    cam = CameraStream(src=args.camera, width=args.width, height=args.height)
    time.sleep(0.5)  # let camera warm up

    print("[3/3] Initializing MediaPipe (lite mode)...")
    extractor = FastKeypointExtractor(complexity=0)

    keypoint_buffer = deque(maxlen=128)
    current_glosses = []
    last_inference_time = 0.0
    inference_interval = 0.8  # run model every 0.8s (it's the heavy part)

    # FPS tracking
    frame_times = deque(maxlen=30)

    print("Demo running! Press Q to quit.")

    while True:
        t0 = time.perf_counter()

        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        # ── MediaPipe extraction (on every frame for smooth tracking) ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps, meta = extractor.extract(rgb)
        keypoint_buffer.append(kps)

        # ── Model inference (throttled) ──
        now = time.perf_counter()
        if now - last_inference_time > inference_interval and len(keypoint_buffer) >= 16:
            last_inference_time = now
            buf = np.array(list(keypoint_buffer))[-64:]
            inp = torch.tensor(buf, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([len(buf)]).to(device)
            with torch.no_grad():
                current_glosses = model.decode(inp, lengths, beam_width=1)[0]

        # ── Draw ──
        draw_skeleton(frame, kps, meta)

        # FPS
        frame_times.append(time.perf_counter() - t0)
        fps = len(frame_times) / max(sum(frame_times), 1e-9)

        draw_hud(frame, current_glosses, fps, len(keypoint_buffer))

        cv2.imshow("ISL Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            keypoint_buffer.clear()
            current_glosses = []

    cam.stop()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
