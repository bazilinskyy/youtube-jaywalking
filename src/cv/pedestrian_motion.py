from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PedestrianMotionExtractor:
    """Extracts pedestrian trajectory and motion features across video clips using ByteTrack."""

    def __init__(
        self,
        yolo_model_path: str = "models/yolo11x.pt",
        conf_thresh: float = 0.30,
        stride: int = 2,
    ) -> None:
        self.model_path = yolo_model_path
        self.conf_thresh = conf_thresh
        self.stride = stride
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def extract(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Tracks pedestrians across the clip and extracts structured motion features."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self._empty_result("none")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if w <= 0:
            cap.release()
            return self._empty_result("none")

        tracks: Dict[int, List[float]] = {}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % self.stride != 0:
                continue

            results = self.model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                classes=[0],  # Person class
                conf=self.conf_thresh,
                verbose=False,
                device=self.device,
            )

            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                tids = results[0].boxes.id.int().cpu().tolist()
                for box, tid in zip(boxes, tids):
                    cx = ((box[0] + box[2]) / 2.0) / max(w, 1)
                    if tid not in tracks:
                        tracks[tid] = []
                    tracks[tid].append(cx)

        cap.release()

        valid_tracks = {tid: xs for tid, xs in tracks.items() if len(xs) >= 3}
        if not valid_tracks:
            return self._empty_result("none")

        # Select primary track by maximum horizontal span (crossing activity)
        primary_tid = max(valid_tracks.keys(), key=lambda tid: max(valid_tracks[tid]) - min(valid_tracks[tid]))
        xs = valid_tracks[primary_tid]
        start_x = xs[0]
        end_x = xs[-1]
        disp = round(abs(end_x - start_x), 2)
        tracked_frames = len(xs)

        pos_start = "curb" if (start_x < 0.25 or start_x > 0.75) else "roadway"
        pos_end = "curb" if (end_x < 0.25 or end_x > 0.75) else "roadway"

        if disp < 0.05:
            movement = "stationary_at_curb" if pos_start == "curb" else "stationary_in_roadway"
        elif pos_start == "curb" and pos_end == "roadway":
            movement = "entering_roadway"
        elif pos_start == "roadway" and pos_end == "curb":
            movement = "exiting_roadway"
        elif pos_start == "roadway" and pos_end == "roadway":
            movement = "crossing_roadway"
        else:
            movement = "moving_along_curb"

        conf = "high" if tracked_frames >= 10 else ("medium" if tracked_frames >= 4 else "low")

        return {
            "position_start": pos_start,
            "position_end": pos_end,
            "start_x": round(start_x, 3),
            "end_x": round(end_x, 3),
            "normalized_displacement": disp,
            "movement": movement,
            "tracking_confidence": conf,
            "tracked_frames": tracked_frames,
            "formatted_context": self._format_context(pos_start, pos_end, disp, movement, conf),
        }

    def _empty_result(self, confidence: str = "none") -> Dict[str, Any]:
        return {
            "position_start": "unknown",
            "position_end": "unknown",
            "start_x": None,
            "end_x": None,
            "normalized_displacement": 0.0,
            "movement": "unknown",
            "tracking_confidence": confidence,
            "tracked_frames": 0,
            "formatted_context": self._format_context("unknown", "unknown", 0.0, "unknown", confidence),
        }

    @staticmethod
    def _format_context(
        pos_start: str, pos_end: str, disp: float, movement: str, conf: str
    ) -> str:
        return (
            f"Pedestrian motion:\n"
            f"position_start: {pos_start}\n"
            f"position_end: {pos_end}\n"
            f"normalized_displacement: {disp:.2f}\n"
            f"movement: {movement}\n"
            f"tracking_confidence: {conf}"
        )
