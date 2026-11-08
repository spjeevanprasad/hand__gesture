# Hand Gesture Recognition (Standalone Project - 60 to 90 FPS)

A real-time, low-latency 21-joint hand skeleton tracker and gesture recognition engine powered by MediaPipe Tasks:
1. **Signal for Help (Domestic Violence & Distress Signal)**:
   - **Step 1**: Palm facing outwards, thumb placed on the palm (4 fingers extended).
   - **Step 2**: Move fingers down over thumb (fist closed over tucked thumb) -> **Triggers "Help"** with glowing emergency alert banner!
2. **Real-Time Finger Counting (0 to 10 Fingers)**:
   - Per-hand finger counter (0 to 5 fingers per hand).
   - Two-hand total finger counter (**10 Fingers / 10 Count** when showing both open hands).
   - Dedicated on-screen floating finger counter widget.
3. **21 3D Joint Tracking**: Tracks full hand articulation with glowing neon skeleton visualization.
4. **Handedness Detection**: Distinguishes Left Hand vs. Right Hand automatically.
5. **Multi-Gesture Recognition**: Recognizes 14+ distinct gestures:
   - **Help** (Emergency Distress Signal)
   - Help Signal (Thumb Tucked - Step 1)
   - Like / Thumbs Up
   - Dislike / Thumbs Down
   - Call Me
   - Rock On / Horns
   - OK Sign
   - Point / One (1)
   - Peace Up / Two Up (2)
   - Peace Down / Two Down (2)
   - Three (3)
   - Four (4)
   - Open Palm / Five (5)
   - Closed Fist (0)
   - Pinch
6. **Anti-Jitter Exponential Smoothing**: Eliminates sensor noise and micro-flicker while maintaining zero latency at 60-90 FPS.

---

## Folder Structure

```
Hand_Gesture_Recognition/
├── assets/
│   ├── clean_preview.jpg
│   └── snapshot_vision_1793775114.jpg
├── models/
│   └── gesture_recognizer.task
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── gesture_classifier.py
│   ├── hand_tracker.py
│   └── visualizer.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Ensure Python 3.10+ is installed. Navigate to this directory and install the required dependencies:

```bash
cd Desktop/Hand_Gesture_Recognition
pip install -r requirements.txt
```

---

## Usage

### Run with Live Webcam
```bash
python main.py
```

### Options & Arguments
- `--cam 0` : Specify webcam device index (default: `0`).
- `--num-hands 2` : Maximum simultaneous hands to track (default: `2`).
- `--min-conf 0.50` : Minimum detection confidence threshold (default: `0.50`).
- `--headless` : Run in headless smoke test mode without opening GUI windows.
- `--max-frames 50` : Process 50 frames and cleanly exit.
- `--save-preview preview.jpg` : Save an annotated snapshot of the hand gesture results.

### Interactive Controls
- `[S]` : Save snapshot of current frame with hand skeletons and gesture tags.
- `[Q]` or `[ESC]` : Quit the application.
