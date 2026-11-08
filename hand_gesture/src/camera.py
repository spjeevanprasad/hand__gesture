import os
import sys
import time
import threading
import cv2
import numpy as np

class ThreadedCamera:
    """
    Robust Asynchronous Threaded Camera Reader:
    - Auto-detects available webcam (tries index 0, then index 1)
    - Sensor warmup loop (handles slow webcam driver initialization)
    - Threaded buffer to eliminate OpenCV I/O blocking
    - Fallback image feed if camera is in use or unavailable
    """
    def __init__(self, src=0, width=640, height=480, is_headless=False, fallback_image=None):
        self.is_headless = is_headless
        self.stopped = False
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.src = src
        self.cap = None

        if is_headless:
            self._init_headless(width, height, fallback_image)
            return

        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened() and src == 0:
            print("[INFO] Camera index 0 not available. Trying camera index 1...")
            self.cap.release()
            self.cap = cv2.VideoCapture(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"[WARN] No hardware webcam accessible. Using test preview feed.")
            self._init_headless(width, height, fallback_image)
            return

        for _ in range(20):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.ret = True
                self.frame = frame
                break
            time.sleep(0.05)

        if not self.ret:
            print("[WARN] Camera opened but delivered empty frames. Using test preview feed.")
            self._init_headless(width, height, fallback_image)
            return

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_headless(self, width, height, fallback_image):
        if fallback_image and os.path.exists(fallback_image):
            img = cv2.imread(fallback_image)
            if img is not None:
                self.frame = cv2.resize(img, (width, height))
                self.ret = True
                return
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(self.frame, (40, 40), (width - 40, height - 40), (35, 35, 35), -1)
        cv2.putText(self.frame, "Hand Gesture Test Feed", (width // 5, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        self.ret = True

    def _capture_loop(self):
        while not self.stopped:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.ret = ret
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.01)

    def isOpened(self):
        return self.ret and self.frame is not None

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.stopped = True
        time.sleep(0.02)
        if self.cap and self.cap.isOpened():
            self.cap.release()
