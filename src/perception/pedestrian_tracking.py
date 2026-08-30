"""Pedestrian pose detection and multi-object tracking using YOLO26x-Pose and BoT-SORT.

This module provides the PedestrianTracker class, which tracks pedestrians across video
sequences, extracts kinematic metrics (lateral displacement, vertical scene coordinate,
and track duration), and identifies the dominant candidate crossing trajectory.
"""

from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO


class PedestrianTracker:
    """Tracks pedestrians across video sequences and extracts dominant crossing candidate trajectories."""

    def __init__(
        self,
        model_path: str = "yolo26x-pose.pt",
        tracker_config: str = "configs/botsort_custom.yaml",
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> None:
        """Initializes the pedestrian tracker with YOLO pose weights and BoT-SORT configuration.

        Args:
            model_path: Filepath or model name for YOLO pose estimation weights.
            tracker_config: Path to the BoT-SORT tracker YAML configuration file.
            conf: Confidence threshold for pedestrian detection filtering.
            iou: Intersection-over-Union threshold for NMS and association.
        """
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou

    def track_video(
        self, video_path: str, fps: float
    ) -> Tuple[float, float, float, float, List[dict]]:
        """Executes multi-object tracking across an entire video and identifies the dominant pedestrian track.

        Identifies candidate pedestrian tracks having at least 3 observed frames, scores them
        by total horizontal normalized lateral displacement (|cx_end - cx_start|), and selects
        the highest-displacement candidate as the primary crossing pedestrian.

        Args:
            video_path: Path to the input video file (.mp4).
            fps: Frame rate of the input video for track duration calculation.

        Returns:
            A tuple containing:
                - lateral_displacement (float): Maximum normalized horizontal displacement (0.0 to 1.0).
                - mean_x (float): Average normalized horizontal position of the bounding box center (0.0 to 1.0).
                - mean_bottom_y (float): Average normalized vertical position of the bounding box base (0.0 to 1.0).
                - track_duration_sec (float): Duration in seconds the dominant pedestrian was actively tracked.
                - dominant_frames (List[dict]): Frame-by-frame bounding box annotations for the dominant track.
        """
        results_track = self.model.track(
            source=video_path,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
        )

        track_boxes: Dict[int, List[dict]] = {}
        for f_i, res in enumerate(results_track):
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int)
                boxes = res.boxes.xywhn.cpu().numpy()
                for tid, box in zip(ids, boxes):
                    if tid not in track_boxes:
                        track_boxes[tid] = []
                    track_boxes[tid].append({
                        "frame": f_i,
                        "cx": float(box[0]),
                        "cy": float(box[1]),
                        "w": float(box[2]),
                        "h": float(box[3]),
                        "by": float(box[1] + box[3] / 2.0),
                    })

        lat_disp = 0.0
        mean_x = 0.50
        mean_y = 0.50
        track_dur = 0.0
        dom_frames: List[dict] = []

        if track_boxes:
            scored = []
            for tid, f_list in track_boxes.items():
                # Filter tracks with at least 3 frames to avoid spurious single-frame detections
                if len(f_list) >= 3:
                    xs = [p["cx"] for p in f_list]
                    disp = abs(xs[-1] - xs[0])
                    scored.append((disp, tid, f_list))
            if scored:
                # Dominant candidate is the pedestrian with the largest transverse trajectory
                scored.sort(key=lambda x: x[0], reverse=True)
                lat_disp, _, dom_frames = scored[0]
                mean_x = float(np.mean([p["cx"] for p in dom_frames]))
                mean_y = float(np.mean([p["by"] for p in dom_frames]))
                track_dur = float(len(dom_frames) / max(1.0, fps))

        return (
            round(lat_disp, 3),
            round(mean_x, 3),
            round(mean_y, 3),
            round(track_dur, 2),
            dom_frames,
        )
