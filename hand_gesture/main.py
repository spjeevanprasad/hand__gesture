"""
Hand Gesture Recognition System (Independent Standalone Application - 60 to 90 FPS)
----------------------------------------------------------------------------------
Features:
- Decoupled real-time vision architecture running at 60 - 90+ FPS
- 21 3D hand landmark tracking using MediaPipe GestureRecognizer
- Left Hand vs. Right Hand classification (Handedness)
- Real-time gesture classification (Thumbs Up/Down, Call Me, Rock, Peace, etc.)
- Neon glowing skeleton overlay with anti-jitter landmark smoothing
"""

import os
import sys

# Auto-delegate to Python 3.13 if invoked with Python 3.14 (where MediaPipe is installed)
if sys.version_info >= (3, 14):
    py313 = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python313\python.exe")
    if os.path.exists(py313) and os.path.abspath(sys.executable).lower() != os.path.abspath(py313).lower():
        import subprocess
        result = subprocess.run([py313] + sys.argv)
        sys.exit(result.returncode)

import argparse
import time
from pathlib import Path
import cv2

# Ensure local src/ is importable
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.hand_tracker import HandTracker
from src.worker import AsyncHandWorker
from src.camera import ThreadedCamera
from src.visualizer import draw_hand_skeleton, draw_gesture_badge, draw_hud


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Hand Gesture Recognition")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--num-hands", type=int, default=2, help="Maximum hands to track (default: 2)")
    parser.add_argument("--min-conf", type=float, default=0.50, help="Tracking confidence threshold (default: 0.50)")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory where model file is stored")
    parser.add_argument("--test-image", type=str, default="assets/snapshot_vision_1793775114.jpg", help="Fallback/test image path")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window for testing")
    parser.add_argument("--max-frames", type=int, default=0, help="Frames before exit (0 for continuous loop)")
    parser.add_argument("--save-preview", type=str, default="", help="Optional snapshot output image path")
    return parser.parse_args()


def main():
    args = parse_args()

    models_path = Path(__file__).parent / args.models_dir
    model_file = models_path / "gesture_recognizer.task"
    test_img = Path(__file__).parent / args.test_image

    print("=" * 66)
    print("  HAND GESTURE RECOGNITION SYSTEM (60-90 FPS)")
    print("=" * 66)
    print(f"  - Camera Index : {args.cam}")
    print(f"  - Target FPS   : 60 - 90 FPS")
    print(f"  - Num Hands    : {args.num_hands} (Up to 10 Fingers Tracking)")
    print(f"  - Min Conf     : {args.min_conf}")
    print(f"  - Distress Sig : 'Help' (Domestic Violence / Distress Signal)")
    print(f"  - Finger Count : Real-Time 0 to 10 Finger Counter")
    print(f"  - Model Path   : {model_file}")
    print(f"  - Headless     : {args.headless}")
    print("  - Controls     : [S] Save Snapshot | [Q]/[ESC] Quit")
    print("=" * 66)

    # 1. Initialize Hand Tracker and Background AI Worker
    tracker = HandTracker(
        model_path=str(model_file),
        num_hands=args.num_hands,
        min_conf=args.min_conf
    )
    worker = AsyncHandWorker(tracker)

    # 2. Initialize High-Speed Camera Feed
    camera = ThreadedCamera(
        src=args.cam,
        width=640,
        height=480,
        is_headless=args.headless,
        fallback_image=str(test_img)
    )

    if not camera.isOpened():
        print(f"[ERROR] Could not start camera at index {args.cam}.")
        sys.exit(1)

    window_name = "Hand Gesture Recognition (Help Signal & 10-Finger Counter)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = None
    fps = 60.0
    frame_count = 0

    print("\n[INFO] Starting high-FPS hand tracking loop...")
    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            frame_count += 1

            # Flip horizontally for natural mirror feel on webcam
            if not args.headless:
                frame = cv2.flip(frame, 1)

            # Update background worker
            worker.update_frame(frame)

            # Retrieve latest hand landmarks instantly
            hands = worker.get_results()

            occupied_badges = []
            active_gestures = []
            total_fingers = 0
            help_active = False
            hand_finger_summaries = []

            for hand_info in hands:
                if len(hand_info) >= 8:
                    px_points, handedness_str, gesture_str, g_conf, box, hand_color, finger_count, is_help = hand_info
                else:
                    px_points, handedness_str, gesture_str, g_conf, box, hand_color = hand_info[:6]
                    finger_count = 0
                    is_help = (gesture_str == "Help")

                total_fingers += finger_count
                if is_help:
                    help_active = True

                short_hand = "R" if "Right" in handedness_str else "L"
                hand_finger_summaries.append(f"{short_hand}: {finger_count}F")
                active_gestures.append(f"{handedness_str}: {gesture_str}")

                # Draw 21-joint neon skeleton (glows crimson red if Help!)
                draw_hand_skeleton(frame, px_points, (0, 0, 255) if is_help else hand_color)

                # Draw gesture & handedness badge
                draw_gesture_badge(frame, hand_info, occupied_badges)

            # Build summary for Finger Count Widget e.g. "Left: 5F | Right: 5F"
            hand_counts_str = " | ".join(hand_finger_summaries)

            # High-precision FPS calculation
            curr_time = time.perf_counter()
            if prev_time is not None:
                dt = curr_time - prev_time
                if dt > 0:
                    curr_fps = 1.0 / dt
                    fps = 0.90 * fps + 0.10 * curr_fps
            prev_time = curr_time

            # Render top HUD banner with Finger Count and Emergency Alert Banner
            draw_hud(frame, len(hands), active_gestures, fps, total_fingers=total_fingers, help_active=help_active, hand_counts_summary=hand_counts_str)

            # Save preview if requested
            if args.save_preview and (frame_count == args.max_frames or frame_count == 1):
                cv2.imwrite(args.save_preview, frame)
                print(f"[INFO] Saved hand gesture preview to: {args.save_preview}")

            # GUI Display
            if not args.headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):
                    print("[INFO] Exit requested by user.")
                    break
                elif key == ord('s') or key == ord('S'):
                    out_name = f"gesture_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(out_name, frame)
                    print(f"[SUCCESS] Snapshot saved to: {out_name}")

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"[INFO] Reached requested max frames ({args.max_frames}). Completed.")
                break

    finally:
        worker.stop()
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] Cleaned up resources. Shutdown complete.")


if __name__ == "__main__":
    main()
