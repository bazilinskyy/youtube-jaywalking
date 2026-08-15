from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class VehicleStateExtractor:
    """Extracts structured vehicle interaction and ego-motion context using YOLO11x + ByteTrack."""

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
        """Analyzes vehicle motion and interaction across video frames."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self._empty("none")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or w <= 0 or h <= 0:
            cap.release()
            return self._empty("none")

        vehicle_tracks: Dict[int, List[Dict[str, float]]] = {}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % self.stride != 0:
                continue

            # Track COCO vehicle classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
            results = self.model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                classes=[2, 3, 5, 7],
                conf=self.conf_thresh,
                verbose=False,
                device=self.device,
            )

            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                tids = results[0].boxes.id.int().cpu().tolist()
                for box, tid in zip(boxes, tids):
                    cx = ((box[0] + box[2]) / 2.0) / max(w, 1)
                    cy = ((box[1] + box[3]) / 2.0) / max(h, 1)
                    area = ((box[2] - box[0]) * (box[3] - box[1])) / max(w * h, 1)
                    if tid not in vehicle_tracks:
                        vehicle_tracks[tid] = []
                    vehicle_tracks[tid].append({"cx": cx, "cy": cy, "area": area, "frame": frame_idx})

        cap.release()

        valid_tracks = {tid: pts for tid, pts in vehicle_tracks.items() if len(pts) >= 3}
        if not valid_tracks:
            return {
                "ego_vehicle_state": "unknown",
                "approaching_vehicle_state": "none_present",
                "interaction": "no_vehicle_conflict",
                "confidence": "high",
                "formatted_context": self._format("unknown", "none_present", "no_vehicle_conflict", "high"),
            }

        # Differentiate lateral/parked vehicles from active roadway lane vehicles
        lateral_tracks = [pts for pts in valid_tracks.values() if all(p["cx"] < 0.25 or p["cx"] > 0.75 for p in pts)]
        roadway_tracks = [pts for pts in valid_tracks.values() if any(0.25 <= p["cx"] <= 0.75 for p in pts)]

        # Estimate ego-vehicle state from background/lateral vehicle motion
        if lateral_tracks:
            lateral_dys = [abs(pts[-1]["cy"] - pts[0]["cy"]) for pts in lateral_tracks]
            avg_lateral_dy = float(np.mean(lateral_dys))
            if avg_lateral_dy < 0.035:
                ego_state = "stopped"
            else:
                ego_state = "moving"
        else:
            ego_state = "unknown"

        if not roadway_tracks:
            appr_state = "none_present"
            interaction = "no_vehicle_conflict"
            confidence = "high"
        else:
            # Analyze roadway vehicle expansion and approach
            active_approaching = 0
            stopped_yielding = 0

            for pts in roadway_tracks:
                areas = [p["area"] for p in pts]
                d_area = (areas[-1] - areas[0]) / max(areas[0], 1e-4)
                dy = pts[-1]["cy"] - pts[0]["cy"]

                if d_area > 0.40 or dy > 0.08:
                    active_approaching += 1
                elif abs(d_area) < 0.15 and abs(dy) < 0.03:
                    stopped_yielding += 1

            if active_approaching > 0:
                appr_state = "moving"
                interaction = "active_traffic"
                confidence = "high"
            elif stopped_yielding > 0 or ego_state == "stopped":
                appr_state = "stopped"
                interaction = "yielding"
                confidence = "high"
            else:
                appr_state = "unknown"
                interaction = "unknown"
                confidence = "medium"

        return {
            "ego_vehicle_state": ego_state,
            "approaching_vehicle_state": appr_state,
            "interaction": interaction,
            "confidence": confidence,
            "formatted_context": self._format(ego_state, appr_state, interaction, confidence),
        }

    def _empty(self, confidence: str) -> Dict[str, Any]:
        return {
            "ego_vehicle_state": "unknown",
            "approaching_vehicle_state": "unknown",
            "interaction": "unknown",
            "confidence": confidence,
            "formatted_context": self._format("unknown", "unknown", "unknown", confidence),
        }

    @staticmethod
    def _format(ego: str, appr: str, inter: str, conf: str) -> str:
        return (
            f"Vehicle interaction:\n"
            f"ego_vehicle_state: {ego}\n"
            f"approaching_vehicle_state: {appr}\n"
            f"interaction: {inter}\n"
            f"confidence: {conf}"
        )
