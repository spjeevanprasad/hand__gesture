import time
import threading

class AsyncHandWorker:
    """
    Decoupled Asynchronous MediaPipe Hand & Gesture Worker:
    Runs MediaPipe Tasks tracking in background thread, allowing the main camera
    and skeleton rendering loop to run smoothly at 60 - 90+ FPS.
    """
    def __init__(self, tracker):
        self.tracker = tracker
        self.lock = threading.Lock()
        self.latest_frame = None
        self.detected_hands = []
        self.running = True

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_results(self):
        with self.lock:
            return list(self.detected_hands)

    def _worker_loop(self):
        while self.running:
            frame_to_proc = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_proc = self.latest_frame
                    self.latest_frame = None

            if frame_to_proc is None:
                time.sleep(0.001)
                continue

            try:
                hands = self.tracker.process(frame_to_proc)
                with self.lock:
                    self.detected_hands = hands
            except Exception:
                pass
            # No sleep after processing - immediately grab next available frame

    def stop(self):
        self.running = False
