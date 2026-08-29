from __future__ import annotations
from typing import List
import numpy as np
import cv2

from src.cv.crosswalk_utils import CrosswalkRegion, merge_regions


class CrosswalkDetector:
    """
    Classical Computer Vision Crosswalk Detector.

    Uses HSV color thresholding, morphological filtering, spatial ROI constraints,
    aspect ratio bounds, white pixel density analysis, and horizontal/vertical
    intensity profile gradient stripe scoring (zebra pattern detection).

    Requires ZERO neural network weights or external models.
    """

    def __init__(self, conf: float = 0.2):
        self.conf = conf

    def detect(self, frame: np.ndarray) -> List[CrosswalkRegion]:
        """Detect crosswalk regions in a single BGR frame."""
        return self._detect_classical(frame)

    def _detect_classical(self, frame: np.ndarray) -> List[CrosswalkRegion]:
        h, w = frame.shape[:2]

        # Convert to HSV and find white/bright road markings
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 40, 255]))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Focus on lower portion of frame where crosswalks appear (y >= 0.50)
        roi_mask = np.zeros_like(white_mask)
        roi_y_start = int(h * 0.50)
        roi_mask[roi_y_start:, :] = 1
        white_mask = white_mask * roi_mask

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        min_area = 0.001 * h * w

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0

            # Crosswalks are wider than tall, roughly rectangular
            if aspect < 0.8 or aspect > 6.0:
                continue

            # Check vertical position (lower portion, but not too close to bottom edge)
            y_center_norm = (y + ch / 2) / h
            y_top_norm = y / h
            if y_center_norm < 0.55 or y_top_norm > 0.92:
                continue

            # Compute white pixel density within the bounding box
            roi_white = white_mask[y:y + ch, x:x + cw]
            white_density = float(np.sum(roi_white > 0)) / max(roi_white.size, 1)

            if white_density < 0.15:
                continue

            # Try stripe analysis on the grayscale patch
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            patch = gray[y:y + ch, x:x + cw]
            stripe_score = self._compute_stripe_score(patch)
            has_zebra_pattern = stripe_score > 0.12

            # Confidence: dense white region with zebra pattern is high confidence
            base_conf = min(1.0, white_density * 2.0)
            pattern_bonus = 0.3 if has_zebra_pattern else 0.0
            position_bonus = min(0.2, (y_center_norm - 0.55) * 2.0)
            conf = min(1.0, base_conf + pattern_bonus + position_bonus)

            candidates.append(CrosswalkRegion(
                x1=round(x / w, 4),
                y1=round(y / h, 4),
                x2=round((x + cw) / w, 4),
                y2=round((y + ch) / h, 4),
                confidence=round(conf, 3),
                method="classical",
            ))

        # Filter low-confidence regions
        candidates = [c for c in candidates if c.confidence > self.conf]
        candidates.sort(key=lambda r: r.confidence, reverse=True)
        return merge_regions(candidates[:3])

    def _compute_stripe_score(self, patch: np.ndarray) -> float:
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            return 0.0

        # Analyze horizontal intensity profile for alternating pattern
        horizontal_profile = np.mean(patch, axis=1)
        grad = np.abs(np.diff(horizontal_profile))
        grad_mean = float(np.mean(grad))

        if grad_mean < 1.0:
            return 0.0

        # Count zero crossings as measure of stripe alternation
        profile_norm = horizontal_profile - np.mean(horizontal_profile)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(profile_norm))) > 0) / 2
        crossing_density = zero_crossings / max(len(profile_norm), 1)

        stripe_quality = 0.0
        if 1 <= crossing_density <= 12:
            stripe_quality = min(1.0, crossing_density / 6.0)

        # Vertical profile analysis
        vertical_profile = np.mean(patch, axis=0)
        vert_grad = np.abs(np.diff(vertical_profile))
        vert_grad_mean = float(np.mean(vert_grad))

        score = (grad_mean / 15.0) * 0.3 + stripe_quality * 0.5 + min(1.0, vert_grad_mean / 15.0) * 0.2
        return min(1.0, max(0.0, score))
