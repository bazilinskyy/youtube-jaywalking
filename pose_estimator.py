from __future__ import annotations
from dataclasses import dataclass
import numpy as np

COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

_KP = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# Crossing heuristic thresholds
_ANKLE_SPREAD_RATIO = 0.25   # ankle x-spread / body height
_SHOULDER_SLOPE_MAX = 0.3    # |dy/dx| of shoulder line (near-horizontal = facing cam)


@dataclass
class Pose:
    track_id: int
    frame: int
    keypoints: np.ndarray          # (17, 3): x, y, conf  — normalized [0,1]
    bbox: tuple[float, float, float, float]  # cx, cy, w, h  — normalized


def _iou(a: tuple, b: tuple) -> float:
    """IoU between two (cx,cy,w,h) normalized boxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class PoseEstimator:
    def __init__(self, model_path: str = 'yolo11x-pose.pt', conf: float = 0.5):
        self._model_path = model_path
        self._conf = conf
        self._model = None  # lazy-loaded on first call

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)

    def estimate(self, frame: np.ndarray, track_id_boxes: list[tuple]) -> list[Pose]:
        """
        Run YOLO pose on frame and match detections to provided track boxes by IoU.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            track_id_boxes: list of (track_id, cx, cy, w, h) — all normalized.

        Returns:
            list of Pose, one per matched track (unmatched tracks are skipped).
        """
        if not track_id_boxes:
            return []

        self._load()
        h, w = frame.shape[:2]

        results = self._model(frame, conf=self._conf, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return []

        # YOLO boxes in xyxy pixel coords → convert to normalized cx,cy,w,h
        det_boxes = []  # (cx,cy,bw,bh) normalized per detection
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            det_boxes.append(((x1 + x2) / (2 * w), (y1 + y2) / (2 * h),
                               (x2 - x1) / w, (y2 - y1) / h))

        kps_all = results.keypoints.data.cpu().numpy()  # (N_det, 17, 3) pixel+conf

        poses: list[Pose] = []
        for track_id, cx, cy, bw, bh in track_id_boxes:
            track_box = (cx, cy, bw, bh)
            best_i, best_iou = -1, 0.0
            for i, db in enumerate(det_boxes):
                iou = _iou(track_box, db)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i < 0 or best_iou < 0.1:
                continue

            kps = kps_all[best_i].copy()          # (17, 3) in pixels
            kps[:, 0] /= w                         # normalize x
            kps[:, 1] /= h                         # normalize y
            poses.append(Pose(
                track_id=track_id,
                frame=0,   # caller should set this; update() in PoseTracker sets it
                keypoints=kps,
                bbox=det_boxes[best_i],
            ))

        return poses

    def is_crossing(self, pose: Pose) -> bool:
        """
        Heuristic: person is crossing (moving laterally) if:
          1. Ankle x-spread / body height > threshold  (side-on silhouette), OR
          2. Shoulder line is near-horizontal (person facing the camera).
        """
        kp = pose.keypoints  # (17,3) normalized x,y,conf

        def valid(idx: int) -> bool:
            return kp[idx, 2] > 0.3

        # --- heuristic 1: ankle spread ---
        if valid(_KP['left_ankle']) and valid(_KP['right_ankle']):
            ankle_spread = abs(kp[_KP['left_ankle'], 0] - kp[_KP['right_ankle'], 0])
            # body height from nose (or top shoulder) to ankle midpoint
            top_y = None
            for idx in (_KP['nose'], _KP['left_shoulder'], _KP['right_shoulder']):
                if valid(idx):
                    top_y = kp[idx, 1]
                    break
            if top_y is not None:
                ankle_y = (kp[_KP['left_ankle'], 1] + kp[_KP['right_ankle'], 1]) / 2
                body_h = abs(ankle_y - top_y)
                if body_h > 0 and ankle_spread / body_h > _ANKLE_SPREAD_RATIO:
                    return True

        # --- heuristic 2: shoulder slope ---
        if valid(_KP['left_shoulder']) and valid(_KP['right_shoulder']):
            dx = kp[_KP['left_shoulder'], 0] - kp[_KP['right_shoulder'], 0]
            dy = kp[_KP['left_shoulder'], 1] - kp[_KP['right_shoulder'], 1]
            if abs(dx) > 1e-6 and abs(dy / dx) < _SHOULDER_SLOPE_MAX:
                return True

        return False
