import os
import math
from pathlib import Path
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .gesture_classifier import evaluate_hand_gesture

class HandTracker:
    """
    MediaPipe Tasks Hand & Gesture Tracking Engine:
    - 21 3D joint landmark coordinates
    - Left Hand vs. Right Hand classification
    - Temporal exponential smoothing on landmarks to eliminate jitter
    """
    def __init__(self, model_path="models/gesture_recognizer.task", num_hands=2, min_conf=0.50):
        self.model_path = str(Path(model_path).resolve())
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MediaPipe gesture model not found at: {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.smoothed_hand_points = {}
        self.last_valid_hands = []
        self.miss_count = 0

    def process(self, frame):
        """
        Processes a BGR camera frame.
        Returns list of tuples:
        (px_points, handedness_str, gesture_str, conf, (hx_min, hy_min, hx_max, hy_max), color)
        """
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = self.recognizer.recognize(mp_image)

        hands_out = []
        if hand_result.hand_landmarks:
            valid_hands = []
            for h_idx in range(len(hand_result.hand_landmarks)):
                lms = hand_result.hand_landmarks[h_idx]
                w_pt = (lms[0].x, lms[0].y)
                is_dup = False
                for v_lms, _ in valid_hands:
                    vw_pt = (v_lms[0].x, v_lms[0].y)
                    if math.hypot(w_pt[0] - vw_pt[0], w_pt[1] - vw_pt[1]) < 0.08:
                        is_dup = True
                        break
                if not is_dup:
                    valid_hands.append((lms, h_idx))

            for lms_tuple in valid_hands:
                landmarks_21, h_idx = lms_tuple

                raw_handedness = hand_result.handedness[h_idx][0].category_name
                # Note: MediaPipe returns inverted handedness for mirrored camera views
                handedness_str = "Right Hand" if raw_handedness == "Left" else "Left Hand"

                raw_px_points = []
                for lm in landmarks_21:
                    px = int(lm.x * frame_w)
                    py = int(lm.y * frame_h)
                    raw_px_points.append((px, py))

                # Temporal smoothing (82% current + 18% previous)
                if handedness_str in self.smoothed_hand_points:
                    prev_pts = self.smoothed_hand_points[handedness_str]
                    smooth_pts = []
                    for p_i in range(21):
                        sx = int(round(0.82 * raw_px_points[p_i][0] + 0.18 * prev_pts[p_i][0]))
                        sy = int(round(0.82 * raw_px_points[p_i][1] + 0.18 * prev_pts[p_i][1]))
                        smooth_pts.append((sx, sy))
                    px_points = smooth_pts
                else:
                    px_points = raw_px_points
                self.smoothed_hand_points[handedness_str] = px_points

                gesture_str, g_conf, finger_count, is_help = evaluate_hand_gesture(landmarks_21, handedness_str)

                hx_min = max(0, min([p[0] for p in px_points]) - 15)
                hy_min = max(0, min([p[1] for p in px_points]) - 25)
                hx_max = min(frame_w, max([p[0] for p in px_points]) + 15)
                hy_max = min(frame_h, max([p[1] for p in px_points]) + 15)

                hand_color = (0, 0, 255) if is_help else ((0, 255, 255) if "Right" in handedness_str else (255, 100, 255))
                hands_out.append((px_points, handedness_str, gesture_str, g_conf,
                                 (hx_min, hy_min, hx_max, hy_max), hand_color, finger_count, is_help))

        # Persistence across brief 3-frame micro-dropouts
        if hands_out:
            self.miss_count = 0
            self.last_valid_hands = hands_out
        else:
            if self.miss_count < 3 and self.last_valid_hands:
                self.miss_count += 1
                hands_out = self.last_valid_hands
            else:
                self.smoothed_hand_points.clear()
                self.last_valid_hands = []

        return hands_out
