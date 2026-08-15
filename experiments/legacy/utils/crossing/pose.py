import numpy as np
import torch
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path: str = "yolo11x-pose.pt") -> None:
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)

    def estimate_pose(self, person_crop: np.ndarray) -> tuple[list[tuple[float, float, float]], str, float]:
        """
        Runs keypoint detection on a cropped pedestrian image.
        Returns:
            - keypoints: list of 17 tuples (x, y, conf)
            - facing_direction: 'FACING_VEHICLE', 'SIDE_VIEW', 'BACK_VIEW', or 'UNKNOWN'
            - stride_ratio: relative horizontal distance between ankles
        """
        empty_kps = [(0.0, 0.0, 0.0)] * 17
        if person_crop is None or person_crop.size == 0:
            return empty_kps, "UNKNOWN", 0.0

        h, w = person_crop.shape[:2]
        if h == 0 or w == 0:
            return empty_kps, "UNKNOWN", 0.0

        # Run pose inference on the pedestrian crop
        results = self.model(person_crop, verbose=False, device=self.device)
        if not results or len(results) == 0 or results[0].keypoints is None:
            return empty_kps, "UNKNOWN", 0.0

        kps_data = results[0].keypoints.data
        if len(kps_data) == 0:
            return empty_kps, "UNKNOWN", 0.0

        # Select the first detected person keypoints (CPU numpy array)
        kps = kps_data[0].cpu().numpy() # shape (17, 3)
        keypoints = [(float(kp[0]), float(kp[1]), float(kp[2])) for kp in kps]

        # Analyze facing direction based on visibility and distance of keypoints
        # 17 keypoints indices:
        # 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear
        # 5: left shoulder, 6: right shoulder, 11: left hip, 12: right hip
        nose_visible = keypoints[0][2] > 0.4
        left_eye_visible = keypoints[1][2] > 0.4
        right_eye_visible = keypoints[2][2] > 0.4
        left_shoulder_visible = keypoints[5][2] > 0.4
        right_shoulder_visible = keypoints[6][2] > 0.4

        facing_direction = "UNKNOWN"
        if left_shoulder_visible and right_shoulder_visible:
            sh_width = abs(keypoints[5][0] - keypoints[6][0])
            # If shoulders are aligned horizontally, they are either facing front or back
            if sh_width > 0.15 * w:
                if nose_visible or left_eye_visible or right_eye_visible:
                    facing_direction = "FACING_VEHICLE"
                else:
                    facing_direction = "BACK_VIEW"
            else:
                facing_direction = "SIDE_VIEW"
        elif left_shoulder_visible or right_shoulder_visible:
            facing_direction = "SIDE_VIEW"

        # Stride ratio: horizontal distance between ankles / crop width
        left_ankle_visible = keypoints[15][2] > 0.4
        right_ankle_visible = keypoints[16][2] > 0.4
        stride_ratio = 0.0
        if left_ankle_visible and right_ankle_visible:
            stride_dist = abs(keypoints[15][0] - keypoints[16][0])
            stride_ratio = float(stride_dist / w)

        return keypoints, facing_direction, stride_ratio
