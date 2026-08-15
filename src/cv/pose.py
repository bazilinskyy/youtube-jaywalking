from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from src.config import get_cv_config


class PoseEstimator:
    """YOLO-Pose keypoint detector for pedestrian facing direction and stride estimation."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        cfg = get_cv_config()
        self.model_path = model_path or cfg.get("pose_model")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path).to(self.device)
        return self._model

    def estimate_pose(
        self, person_crop: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float]], str, float]:
        """
        Runs keypoint detection on a cropped pedestrian image.
        Returns:
            - keypoints: list of 17 tuples (x, y, conf)
            - facing_direction: 'FACING_VEHICLE', 'SIDE_VIEW', 'BACK_VIEW', or 'UNKNOWN'
            - stride_ratio: horizontal distance between ankles / crop width
        """
        empty_kps = [(0.0, 0.0, 0.0)] * 17
        if person_crop is None or person_crop.size == 0:
            return empty_kps, "UNKNOWN", 0.0

        h, w = person_crop.shape[:2]
        if h == 0 or w == 0:
            return empty_kps, "UNKNOWN", 0.0

        results = self.model(person_crop, verbose=False, device=self.device)
        if not results or len(results) == 0 or results[0].keypoints is None:
            return empty_kps, "UNKNOWN", 0.0

        kps_data = results[0].keypoints.data
        if len(kps_data) == 0:
            return empty_kps, "UNKNOWN", 0.0

        kps = kps_data[0].cpu().numpy()
        keypoints = [(float(kp[0]), float(kp[1]), float(kp[2])) for kp in kps]

        nose_vis = keypoints[0][2] > 0.4
        left_eye_vis = keypoints[1][2] > 0.4
        right_eye_vis = keypoints[2][2] > 0.4
        left_sh_vis = keypoints[5][2] > 0.4
        right_sh_vis = keypoints[6][2] > 0.4

        facing = "UNKNOWN"
        if left_sh_vis and right_sh_vis:
            sh_width = abs(keypoints[5][0] - keypoints[6][0])
            if sh_width > 0.15 * w:
                if nose_vis or left_eye_vis or right_eye_vis:
                    facing = "FACING_VEHICLE"
                else:
                    facing = "BACK_VIEW"
            else:
                facing = "SIDE_VIEW"
        elif left_sh_vis or right_sh_vis:
            facing = "SIDE_VIEW"

        left_ankle_vis = keypoints[15][2] > 0.4
        right_ankle_vis = keypoints[16][2] > 0.4
        stride_ratio = 0.0
        if left_ankle_vis and right_ankle_vis:
            stride_dist = abs(keypoints[15][0] - keypoints[16][0])
            stride_ratio = float(stride_dist / max(w, 1))

        return keypoints, facing, stride_ratio
