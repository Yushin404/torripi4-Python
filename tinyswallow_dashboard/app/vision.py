from __future__ import annotations
import time
import threading
import numpy as np
import cv2
from ultralytics import YOLO
from .state import STATE


class Vision:
    def __init__(self, model_path: str, target_class_id: int = 0):
        self.model = YOLO(model_path)
        self.target_class_id = target_class_id
        self._last_frame = None
        self._last_annotated = None
        self._lock = threading.Lock()
        self._running = True

    def stop(self):
        self._running = False

    def set_latest_frame(self, frame: np.ndarray):
        with self._lock:
            self._last_frame = frame

    def get_latest_annotated(self) -> np.ndarray | None:
        with self._lock:
            return None if self._last_annotated is None else self._last_annotated.copy()

    def start_infer_loop(self, fps_limit: float = 15.0) -> threading.Thread:
        interval = 1.0 / max(1.0, fps_limit)

        def loop():
            while self._running and STATE.running:
                t0 = time.time()

                with self._lock:
                    frame = None if self._last_frame is None else self._last_frame.copy()

                if frame is None:
                    time.sleep(0.01)
                    continue

                results = self.model(frame)
                r0 = results[0]
                boxes = r0.boxes

                # target detect
                detected = False
                center_x = None
                size1 = None

                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == self.target_class_id:
                            x1, y1, x2, y2 = box.xyxy[0]
                            center_x = int((x1 + x2) / 2)
                            detected = True
                            size1 = (y2 - y1)*(x2-x1)
                            break

                STATE.target_detected = detected
                STATE.target_center_x = center_x
                STATE.size = size1

                annotated = r0.plot()  # BGR image

                with self._lock:
                    self._last_annotated = annotated

                dt = time.time() - t0
                if dt < interval:
                    time.sleep(interval - dt)

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        return th


def bgr_to_jpeg_bytes(bgr: np.ndarray, width: int = 500, height: int = 500) -> bytes:
    img = cv2.resize(bgr, (width, height))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return b""
    return buf.tobytes()
