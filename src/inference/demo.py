#!/usr/bin/env python
"""
Real-time Webcam Demo for ISL Translation.

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.keypoint_extractor import KeypointExtractor
from src.models.isl_model import ISLTranslator


class WebcamDemo:
    """Real-time sign language translation from webcam."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        window_size: int = 64,
        overlap: int = 32,
        beam_width: int = 5,
    ):
        self.device = device
        self.window_size = window_size
        self.overlap = overlap
        self.beam_width = beam_width
        
        # Load model
        print("Loading model...")
        self.model = ISLTranslator.load(checkpoint_path, use_translation=False)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize keypoint extractor
        self.extractor = KeypointExtractor(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Buffer for keypoints
        self.keypoint_buffer = []
        
        print("Demo ready!")
    
    def process_frame(self, frame: np.ndarray):
        """Extract keypoints from frame and add to buffer."""
        keypoints, metadata = self.extractor.extract_frame(frame)
        self.keypoint_buffer.append(keypoints)
        
        # Keep buffer at window_size
        if len(self.keypoint_buffer) > self.window_size * 2:
            self.keypoint_buffer = self.keypoint_buffer[-self.window_size:]
        
        return keypoints, metadata
    
    def translate(self):
        """Translate current buffer."""
        if len(self.keypoint_buffer) < 16:
            return [], ""
        
        # Prepare input
        keypoints = np.array(self.keypoint_buffer[-self.window_size:])
        keypoints = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(keypoints[0])]).to(self.device)
        
        # Translate
        with torch.no_grad():
            glosses, translations = self.model.translate(
                keypoints, lengths,
                beam_width=self.beam_width,
            )
        
        return glosses[0], translations[0]
    
    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, metadata: dict):
        """Draw keypoints on frame."""
        h, w = frame.shape[:2]
        
        # Draw pose (green)
        if metadata["pose_detected"]:
            for i in range(33):
                x, y, conf = keypoints[i]
                if conf > 0.5:
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        
        # Draw hands (red for left, yellow for right)
        hand_start = 33 + 468  # After pose and face
        
        if metadata["left_hand_detected"]:
            for i in range(21):
                x, y, _ = keypoints[hand_start + i]
                if x > 0 or y > 0:
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        
        if metadata["right_hand_detected"]:
            for i in range(21):
                x, y, _ = keypoints[hand_start + 21 + i]
                if x > 0 or y > 0:
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        
        return frame
    
    def run(self, camera_id: int = 0):
        """Run the demo."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Cannot open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Press 'q' to quit, 'c' to clear buffer")
        
        current_glosses = []
        current_translation = ""
        last_translate_time = 0
        translate_interval = 0.5  # Translate every 0.5 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            keypoints, metadata = self.process_frame(frame)
            
            # Translate periodically
            current_time = time.time()
            if current_time - last_translate_time > translate_interval:
                current_glosses, current_translation = self.translate()
                last_translate_time = current_time
            
            # Draw visualization
            frame = self.draw_keypoints(frame, keypoints, metadata)
            
            # Draw text overlay
            y_offset = 30
            cv2.putText(frame, f"Glosses: {' '.join(current_glosses)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            cv2.putText(frame, f"Translation: {current_translation}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Buffer: {len(self.keypoint_buffer)} frames", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Show frame
            cv2.imshow("ISL Translator", frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.keypoint_buffer.clear()
                current_glosses = []
                current_translation = ""
        
        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()


def main():
    parser = argparse.ArgumentParser(description="ISL Translator Webcam Demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--window", type=int, default=64, help="Sliding window size")
    parser.add_argument("--beam", type=int, default=5, help="Beam search width")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    demo = WebcamDemo(
        checkpoint_path=args.checkpoint,
        device=device,
        window_size=args.window,
        beam_width=args.beam,
    )
    
    demo.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
