from __future__ import annotations
import time
import threading
import numpy as np
import cv2
from ultralytics import YOLO
from .state import STATE


class Vision:
    def __init__(self, model_path: str, priority_class_ids: list[int] | None = None):
        self.model = YOLO(model_path)

        # 追跡の優先順位（後方互換 target 用）
        # 0 = clip, 1 = cone, 2 = stool
        self.priority_class_ids = priority_class_ids or [0, 2]

        self._last_frame = None
        self._last_annotated = None
        self._lock = threading.Lock()
        self._running = True

        # -----------------------------
        # temporal filter parameters
        # -----------------------------
        self.clip_confirm_frames = 2
        self.stool_confirm_frames = 2
        self.cone_confirm_frames = 1  # coneは即反応

        self.clip_hold_frames = 3
        self.stool_hold_frames = 3
        self.cone_hold_frames = 1

        # -----------------------------
        # detection counters
        # -----------------------------
        self.clip_seen_count = 0
        self.stool_seen_count = 0
        self.cone_seen_count = 0

        self.clip_lost_count = 0
        self.stool_lost_count = 0
        self.cone_lost_count = 0

        # -----------------------------
        # last stable detections
        # -----------------------------
        self.clip_stable = None   # (cls, conf, area, x1, y1, x2, y2)
        self.stool_stable = None
        self.cone_stable = None

    def stop(self):
        self._running = False

    def set_latest_frame(self, frame: np.ndarray):
        with self._lock:
            self._last_frame = frame

    def get_latest_annotated(self) -> np.ndarray | None:
        with self._lock:
            return None if self._last_annotated is None else self._last_annotated.copy()

    def _pick_best_of_class(self, det_list, class_id: int):
        cand = [d for d in det_list if d[0] == class_id]
        if not cand:
            return None
        # 同一クラス複数なら confidence 最大
        return max(cand, key=lambda d: d[1])

    def _apply_temporal_filter(
        self,
        best_det,
        seen_count_attr: str,
        lost_count_attr: str,
        stable_attr: str,
        confirm_frames: int,
        hold_frames: int,
    ):
        """
        best_det:
            None or (cls, conf, area, x1, y1, x2, y2)

        returns:
            stable detection tuple or None
        """
        seen_count = getattr(self, seen_count_attr)
        lost_count = getattr(self, lost_count_attr)
        stable_det = getattr(self, stable_attr)

        if best_det is not None:
            seen_count += 1
            lost_count = 0

            # 条件成立で stable 更新
            if seen_count >= confirm_frames:
                stable_det = best_det
        else:
            seen_count = 0
            lost_count += 1

            # しばらくは前回 stable を保持
            if lost_count > hold_frames:
                stable_det = None

        setattr(self, seen_count_attr, seen_count)
        setattr(self, lost_count_attr, lost_count)
        setattr(self, stable_attr, stable_det)

        return stable_det

    def _write_state_from_det(self, prefix: str, det):
        """
        prefix in {"clip", "cone", "stool"}
        """
        if det is None:
            setattr(STATE, f"{prefix}_detected", False)
            setattr(STATE, f"{prefix}_center_x", None)
            setattr(STATE, f"{prefix}_size", None)
            return

        _, conf, area, x1, y1, x2, y2 = det
        setattr(STATE, f"{prefix}_detected", True)
        setattr(STATE, f"{prefix}_center_x", int((x1 + x2) / 2))
        setattr(STATE, f"{prefix}_size", int(area))

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

                det_list = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0]) if box.conf is not None else 0.0
                        x1, y1, x2, y2 = box.xyxy[0]
                        area = float((y2 - y1) * (x2 - x1))
                        det_list.append((cls, conf, area, float(x1), float(y1), float(x2), float(y2)))

                # 0 = clip, 1 = cone, 2 = stool
                best_clip = self._pick_best_of_class(det_list, 0)
                best_cone = self._pick_best_of_class(det_list, 1)
                best_stool = self._pick_best_of_class(det_list, 2)

                # temporal filter
                stable_clip = self._apply_temporal_filter(
                    best_det=best_clip,
                    seen_count_attr="clip_seen_count",
                    lost_count_attr="clip_lost_count",
                    stable_attr="clip_stable",
                    confirm_frames=self.clip_confirm_frames,
                    hold_frames=self.clip_hold_frames,
                )

                stable_cone = self._apply_temporal_filter(
                    best_det=best_cone,
                    seen_count_attr="cone_seen_count",
                    lost_count_attr="cone_lost_count",
                    stable_attr="cone_stable",
                    confirm_frames=self.cone_confirm_frames,
                    hold_frames=self.cone_hold_frames,
                )

                stable_stool = self._apply_temporal_filter(
                    best_det=best_stool,
                    seen_count_attr="stool_seen_count",
                    lost_count_attr="stool_lost_count",
                    stable_attr="stool_stable",
                    confirm_frames=self.stool_confirm_frames,
                    hold_frames=self.stool_hold_frames,
                )

                # state 書き込み
                self._write_state_from_det("clip", stable_clip)
                self._write_state_from_det("cone", stable_cone)
                self._write_state_from_det("stool", stable_stool)

                # 後方互換 target:
                # clip優先、なければstool
                if STATE.clip_detected:
                    STATE.target_detected = True
                    STATE.target_center_x = STATE.clip_center_x
                    STATE.size = STATE.clip_size
                    STATE.target_class_id = 0
                elif STATE.stool_detected:
                    STATE.target_detected = True
                    STATE.target_center_x = STATE.stool_center_x
                    STATE.size = STATE.stool_size
                    STATE.target_class_id = 2
                else:
                    STATE.target_detected = False
                    STATE.target_center_x = None
                    STATE.size = None
                    STATE.target_class_id = None

                STATE.telemetry = {
                    "clip": {
                        "detected": STATE.clip_detected,
                        "center_x": STATE.clip_center_x,
                        "size": STATE.clip_size,
                        "seen_count": self.clip_seen_count,
                        "lost_count": self.clip_lost_count,
                    },
                    "cone": {
                        "detected": STATE.cone_detected,
                        "center_x": STATE.cone_center_x,
                        "size": STATE.cone_size,
                        "seen_count": self.cone_seen_count,
                        "lost_count": self.cone_lost_count,
                    },
                    "stool": {
                        "detected": STATE.stool_detected,
                        "center_x": STATE.stool_center_x,
                        "size": STATE.stool_size,
                        "seen_count": self.stool_seen_count,
                        "lost_count": self.stool_lost_count,
                    },
                    "target_class_id": STATE.target_class_id,
                }

                annotated = r0.plot()

                # デバッグ文字を載せたいならここで追加できる
                cv2.putText(
                    annotated,
                    f"clip:{STATE.clip_detected} stool:{STATE.stool_detected} cone:{STATE.cone_detected}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

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
