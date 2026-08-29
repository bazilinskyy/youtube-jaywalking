"""
Pedestrian pose detection and multi-object tracking using YOLO26x-Pose and BoT-SORT.
"""

from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO


class PedestrianTracker:
    """
    Tracks pedestrians across video sequences and extracts dominant crossing candidate trajectories.
    """

    def __init__(
        self,
        model_path: str = "yolo26x-pose.pt",
        tracker_config: str = "configs/botsort_custom.yaml",
        conf: float = 0.25,
        iou: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou

    def track_video(self, video_path: str, fps: float) -> Tuple[float, float, float, List[dict]]:
        """
        Executes tracking on the input video and identifies the dominant candidate.
        Returns: (lateral_displacement, mean_bottom_y, track_duration_sec, dominant_frames)
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
        mean_y = 0.50
        track_dur = 0.0
        dom_frames: List[dict] = []

        if track_boxes:
            scored = []
            for tid, f_list in track_boxes.items():
                if len(f_list) >= 3:
                    xs = [p["cx"] for p in f_list]
                    disp = abs(xs[-1] - xs[0])
                    scored.append((disp, tid, f_list))
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                lat_disp, _, dom_frames = scored[0]
                mean_y = float(np.mean([p["by"] for p in dom_frames]))
                track_dur = float(len(dom_frames) / max(1.0, fps))

        return round(lat_disp, 3), round(mean_y, 3), round(track_dur, 2), dom_frames
