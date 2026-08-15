from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import ROOT_DIR

_TLD_WEIGHTS_PATH = ROOT_DIR / "models" / "traffic_lights_yolov8x.pt"
_YOLO_FALLBACK_PATH = ROOT_DIR / "models" / "yolo11x.pt"


class TrafficLightClassifier:
    """Traffic light detector and state classifier with temporal smoothing."""

    _model = None
    _is_tld_custom = False
    _history: Dict[int, List[str]] = {}
    TEMPORAL_WINDOW = 5

    @classmethod
    def _get_model(cls) -> YOLO:
        if cls._model is None:
            device = 0 if torch.cuda.is_available() else "cpu"
            if _TLD_WEIGHTS_PATH.exists():
                cls._model = YOLO(str(_TLD_WEIGHTS_PATH)).to(device)
                cls._is_tld_custom = True
            else:
                cls._model = YOLO(str(_YOLO_FALLBACK_PATH)).to(device)
                cls._is_tld_custom = False
        return cls._model

    @classmethod
    def _classify_crop_color(cls, crop: np.ndarray) -> str:
        """Heuristic color estimation for a cropped traffic light box."""
        if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 3:
            return "UNKNOWN"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Red ranges (wraps around 0/180)
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_mask = red_mask1 | red_mask2

        # Green range
        green_mask = cv2.inRange(hsv, np.array([35, 70, 50]), np.array([85, 255, 255]))

        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        total_pixels = max(crop.shape[0] * crop.shape[1], 1)

        red_ratio = red_pixels / total_pixels
        green_ratio = green_pixels / total_pixels

        if red_ratio > 0.05 and red_ratio > green_ratio * 1.5:
            return "RED"
        if green_ratio > 0.05 and green_ratio > red_ratio * 1.5:
            return "GREEN"
        return "UNKNOWN"

    @classmethod
    def detect_from_frame(cls, frame: np.ndarray, conf_thresh: float = 0.25) -> str:
        """Detects traffic lights in a frame and returns RED, GREEN, or UNKNOWN."""
        if frame is None or frame.size == 0:
            return "UNKNOWN"

        model = cls._get_model()
        device = 0 if torch.cuda.is_available() else "cpu"

        if cls._is_tld_custom:
            results = model.predict(frame, verbose=False, conf=conf_thresh, imgsz=640, device=device)
            if not results or len(results) == 0 or results[0].boxes is None:
                return "UNKNOWN"
            
            # Map 20-class TLD output
            red_classes = {1, 6, 7, 8, 9, 12, 17, 18}
            green_classes = {0, 4, 13, 15, 16}
            best_state = "UNKNOWN"
            best_conf = 0.0
            for box in results[0].boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > best_conf:
                    if cid in red_classes:
                        best_state = "RED"
                        best_conf = conf
                    elif cid in green_classes:
                        best_state = "GREEN"
                        best_conf = conf
            return cls._apply_smoothing(-1, best_state)
        else:
            # Fallback using standard COCO YOLO: class 9 is traffic light
            results = model.predict(frame, verbose=False, conf=conf_thresh, classes=[9], device=device)
            if not results or len(results) == 0 or results[0].boxes is None:
                return cls._apply_smoothing(-1, "UNKNOWN")

            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) == 0:
                return cls._apply_smoothing(-1, "UNKNOWN")

            h, w = frame.shape[:2]
            best_state = "UNKNOWN"
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                state = cls._classify_crop_color(crop)
                if state != "UNKNOWN":
                    best_state = state
                    break

            return cls._apply_smoothing(-1, best_state)

    @classmethod
    def _apply_smoothing(cls, track_id: int, state: str) -> str:
        if state == "UNKNOWN":
            return "UNKNOWN"
        if track_id not in cls._history:
            cls._history[track_id] = []
        cls._history[track_id].append(state)
        if len(cls._history[track_id]) > cls.TEMPORAL_WINDOW:
            cls._history[track_id].pop(0)
        counts = Counter(cls._history[track_id])
        return counts.most_common(1)[0][0]

    @classmethod
    def reset(cls) -> None:
        cls._history.clear()
