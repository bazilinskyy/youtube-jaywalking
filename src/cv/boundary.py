from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class RoadBoundary:
    left: float
    right: float
    confidence: float


DEFAULT_LEFT = 0.30
DEFAULT_RIGHT = 0.70


class BoundaryDetector:
    """Detect road boundaries using lane markings, Canny edge detection, and Hough transforms."""

    def __init__(
        self,
        min_road_width: float = 0.30,
        max_road_width: float = 0.85,
        sample_frames: int = 10,
    ):
        self.min_road_width = min_road_width
        self.max_road_width = max_road_width
        self.sample_frames = sample_frames

    def detect(self, video_path: Union[str, Path]) -> RoadBoundary:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return RoadBoundary(left=DEFAULT_LEFT, right=DEFAULT_RIGHT, confidence=0.0)

        sample_indices = np.linspace(0, total_frames - 1, self.sample_frames, dtype=int)
        left_votes, right_votes = [], []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            left, right = self._detect_frame_boundaries(frame)
            if left is not None and right is not None:
                left_votes.append(left)
                right_votes.append(right)

        cap.release()

        if left_votes:
            left = float(np.median(left_votes))
            right = float(np.median(right_votes))
            confidence = min(1.0, len(left_votes) / self.sample_frames)
        else:
            left = DEFAULT_LEFT
            right = DEFAULT_RIGHT
            confidence = 0.0

        # Enforce min/max width
        width = right - left
        if width < self.min_road_width:
            center = (left + right) / 2
            left = center - self.min_road_width / 2
            right = center + self.min_road_width / 2
        if width > self.max_road_width:
            center = (left + right) / 2
            left = center - self.max_road_width / 2
            right = center + self.max_road_width / 2

        left = float(max(0.0, min(1.0, left)))
        right = float(max(0.0, min(1.0, right)))
        return RoadBoundary(left=left, right=right, confidence=confidence)

    def _detect_frame_boundaries(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[int(h * 0.55):int(h * 0.92), :]
        roi_h, roi_w = roi.shape[:2]

        edges = cv2.Canny(roi, 40, 120)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=20,
            minLineLength=int(roi_w * 0.08),
            maxLineGap=20,
        )
        if lines is None:
            return None, None

        vert_lines, horiz_centers = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            xc = (x1 + x2) / 2 / w
            if angle > 60:
                vert_lines.append(xc)
            elif angle < 25:
                horiz_centers.append(xc)

        if len(vert_lines) >= 2:
            vert_lines.sort()
            left = float(np.percentile(vert_lines, 15))
            right = float(np.percentile(vert_lines, 85))
        elif len(horiz_centers) >= 2:
            horiz_centers.sort()
            left = float(np.percentile(horiz_centers, 10))
            right = float(np.percentile(horiz_centers, 90))
        else:
            return None, None

        width = right - left
        if width < self.min_road_width or width > self.max_road_width:
            return None, None

        return left, right


def get_pedestrian_spatial_position(
    frame: np.ndarray,
    boundary: RoadBoundary,
    yolo_model: Optional[object] = None,
) -> str:
    """
    Determines whether the primary pedestrian in the frame is on the
    sidewalk, at the curb, inside the active roadway, or uncertain.
    """
    if yolo_model is None:
        return "uncertain"

    try:
        results = yolo_model(frame, classes=[0], verbose=False)[0]
    except Exception:
        return "uncertain"

    if len(results.boxes) == 0:
        return "uncertain"

    h, w = frame.shape[:2]
    # Find the most prominent pedestrian (largest area in bottom half)
    best_box = None
    best_area = 0.0
    for b in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = b
        area = (x2 - x1) * (y2 - y1)
        yc = (y1 + y2) / (2 * h)
        if yc >= 0.35 and area > best_area:
            best_area = area
            best_box = b

    if best_box is None:
        return "uncertain"

    x1, y1, x2, y2 = best_box
    xc = (x1 + x2) / (2 * w)

    curb_margin = 0.06
    left_bound = boundary.left
    right_bound = boundary.right

    if xc < (left_bound - curb_margin) or xc > (right_bound + curb_margin):
        return "sidewalk"
    elif abs(xc - left_bound) <= curb_margin or abs(xc - right_bound) <= curb_margin:
        return "curb"
    elif (left_bound + curb_margin) < xc < (right_bound - curb_margin):
        return "roadway"
    else:
        return "curb"
