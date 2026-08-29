import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import torch
from ultralytics import YOLO

from src.config import get_cv_config
from src.cv.traffic_light import TrafficLightClassifier


class CVJaywalkingDetector:
    """Classical computer vision baseline using YOLO11x tracking and heuristic signal fusion."""

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        conf_thresh: Optional[float] = None,
    ) -> None:
        cfg = get_cv_config()
        self.model_path = yolo_model_path or cfg.get("yolo_model")
        self.conf_thresh = conf_thresh or cfg.get("confidence_threshold", 0.5)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path).to(self.device)
        return self._model

    def predict(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Runs pedestrian tracking and signal checks across the video."""
        t0 = time.time()
        TrafficLightClassifier.reset()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "prediction": "unknown",
                "confidence": "low",
                "reason": f"Cannot open video: {video_path}",
                "elapsed_seconds": round(time.time() - t0, 3),
            }

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_idx = 0
        person_tracks = defaultdict(list)
        tl_states: List[str] = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1

            # Traffic light check
            tl_state = TrafficLightClassifier.detect_from_frame(frame)
            if tl_state != "UNKNOWN":
                tl_states.append(tl_state)

            # YOLO pedestrian tracking (class 0)
            results = self.model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=self.conf_thresh,
                verbose=False,
                device=self.device,
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                tids = results[0].boxes.id
                track_ids = tids.int().cpu().tolist() if tids is not None else [-1] * len(classes)

                for box, c, tid in zip(boxes, classes, track_ids):
                    if int(c) == 0:  # Pedestrian
                        cx = ((box[0] + box[2]) / 2) / max(w, 1)
                        person_tracks[tid].append(cx)

        cap.release()

        # Crossing evaluation
        has_crossing = False
        for tid, xs in person_tracks.items():
            if len(xs) < 3:
                continue
            x_range = max(xs) - min(xs)
            road_frames = sum(1 for x in xs if 0.15 < x < 0.85)
            if x_range > 0.10 and road_frames >= 2:
                has_crossing = True
                break

        tl_counts = Counter(tl_states)
        red_count = tl_counts.get("RED", 0)
        green_count = tl_counts.get("GREEN", 0)
        colored = red_count + green_count
        red_ratio = red_count / colored if colored > 0 else 0.0

        if not has_crossing:
            pred = "compliant"
            reason = "No crossing motion detected on road"
            conf = "high"
        elif red_count > 0 and red_ratio > 0.15:
            pred = "jaywalking"
            reason = f"Signal violation: RED light detected (red ratio {red_ratio:.1%})"
            conf = "medium"
        elif green_count >= 5:
            pred = "compliant"
            reason = f"Legal crossing: GREEN light detected ({green_count} frames)"
            conf = "medium"
        else:
            pred = "jaywalking"
            reason = "Unsignalized crossing without confirmed green signal"
            conf = "low"

        return {
            "prediction": pred,
            "confidence": conf,
            "reason": reason,
            "total_frames": frame_idx,
            "persons_tracked": len(person_tracks),
            "has_crossing": has_crossing,
            "tl_counts": dict(tl_counts),
            "red_ratio": round(red_ratio, 2),
            "elapsed_seconds": round(time.time() - t0, 3),
        }
