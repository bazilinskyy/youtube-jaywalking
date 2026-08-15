import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# TLD-READY YOLOv8x model — 20 traffic light classes
# Class mapping to simple RED/YELLOW/GREEN:
#   RED:    circle_red, arrow_left_red, arrow_right_red, arrow_straight_red,
#           arrow_left_red_yellow, arrow_right_red_yellow, arrow_straight_red_yellow,
#           arrow_straight_left_red
#   YELLOW: circle_yellow, circle_red_yellow, arrow_left_yellow, arrow_right_yellow,
#           arrow_straight_yellow, arrow_straight_left_yellow
#   GREEN:  circle_green, arrow_left_green, arrow_right_green, arrow_straight_green,
#           arrow_straight_left_green

_RED_CLASSES = {1, 6, 7, 8, 9, 12, 17, 18}
_YELLOW_CLASSES = {3, 5, 10, 11, 14, 19}
_GREEN_CLASSES = {0, 4, 13, 15, 16}
_OFF_CLASSES = {2}

_WEIGHTS_PATH = str(Path(__file__).resolve().parents[2] / "yolo11x.pt")  # fallback
_TLD_WEIGHTS_URL = "/tmp/tld_ready/model_weights/traffic_lights_yolov8x.pt"
_MODEL = None
_HISTORY = {}
TEMPORAL_WINDOW = 5


def _get_model():
    global _MODEL
    if _MODEL is None:
        weights = _TLD_WEIGHTS_URL if Path(_TLD_WEIGHTS_URL).exists() else _WEIGHTS_PATH
        _MODEL = YOLO(weights)
    return _MODEL


def _map_class(cls_id: int) -> str:
    """Map 20-class TLD output to simple RED/YELLOW/GREEN."""
    if cls_id in _RED_CLASSES:
        return "RED"
    if cls_id in _YELLOW_CLASSES:
        return "YELLOW"
    if cls_id in _GREEN_CLASSES:
        return "GREEN"
    if cls_id in _OFF_CLASSES:
        return "OFF"
    return "UNKNOWN"


class TrafficLightClassifier:
    """Drop-in replacement for old CNN classifier. Uses TLD-READY YOLOv8x."""

    @classmethod
    def classify_state(cls, traffic_light_crop: np.ndarray, track_id: int = -1) -> str:
        """
        Classify a traffic light crop using TLD-READY model.
        Returns RED/YELLOW/GREEN/UNKNOWN with temporal smoothing.
        """
        if traffic_light_crop is None or traffic_light_crop.size == 0:
            return "UNKNOWN"
        h, w = traffic_light_crop.shape[:2]
        if h < 6 or w < 2:
            return "UNKNOWN"

        model = _get_model()
        results = model.predict(traffic_light_crop, verbose=False, conf=0.3, imgsz=64)
        return cls._process_results(results, track_id)

    @classmethod
    def detect_from_frame(cls, frame: np.ndarray) -> str:
        """
        Run TLD-READY on a full frame to detect traffic lights.
        Independent of main YOLO — finds TLs that main model misses.
        Returns most prevalent RED/YELLOW/GREEN state, or UNKNOWN.
        """
        if frame is None or frame.size == 0:
            return "UNKNOWN"

        model = _get_model()
        results = model.predict(frame, verbose=False, conf=0.25, imgsz=640)
        return cls._process_results(results, track_id=-1)

    @classmethod
    def _process_results(cls, results, track_id=-1) -> str:
        """Shared logic: extract best TL state from YOLO results."""
        if not results or len(results) == 0:
            return "UNKNOWN"

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return "UNKNOWN"

        best_conf = 0
        best_state = "UNKNOWN"
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            state = _map_class(cls_id)
            if conf > best_conf and state not in ("OFF", "UNKNOWN"):
                best_conf = conf
                best_state = state

        if best_state == "UNKNOWN":
            return "UNKNOWN"

        # Temporal smoothing
        if track_id not in _HISTORY:
            _HISTORY[track_id] = []
        _HISTORY[track_id].append(best_state)
        if len(_HISTORY[track_id]) > TEMPORAL_WINDOW:
            _HISTORY[track_id].pop(0)

        from collections import Counter
        counts = Counter(_HISTORY[track_id])
        return counts.most_common(1)[0][0]

    @classmethod
    def reset(cls):
        """Clear temporal history."""
        _HISTORY.clear()

