"""Semantic road surface segmentation using SegFormer-B0.

This module provides the RoadSegmenter class, which leverages a lightweight SegFormer-B0
model fine-tuned on Cityscapes to segment drivable road surfaces and evaluate whether
pedestrian foot positions overlap with the active roadway.
"""

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class RoadSegmenter:
    """SegFormer-B0 Cityscapes segmentation model for identifying drivable roadway pixels."""

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
        device: str = "cuda",
    ) -> None:
        """Initializes the SegFormer road segmenter.

        Args:
            model_name: HuggingFace model identifier for pretrained SegFormer weights.
            device: Target computing device ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def segment_road_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Segments the drivable road mask from an input BGR image.

        Args:
            image_bgr: Input image in OpenCV BGR format (H, W, 3).

        Returns:
            A binary uint8 2D array of shape (H, W) where 1 indicates road (Cityscapes class 0).
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        # Cityscapes: Class 0 corresponds to drivable 'road'
        road_mask = (pred_seg == 0).astype(np.uint8)
        return road_mask

    def evaluate_foot_road_overlap(
        self,
        road_mask: np.ndarray,
        foot_x_norm: float,
        foot_y_norm: float,
        radius_px: int = 24,
    ) -> float:
        """Calculates the ratio of road pixels within a circular radius of the pedestrian base coordinates.

        Args:
            road_mask: Binary road segmentation mask (1=road, 0=non-road).
            foot_x_norm: Normalized horizontal coordinate of pedestrian base (0.0 to 1.0).
            foot_y_norm: Normalized vertical coordinate of pedestrian base (0.0 to 1.0).
            radius_px: Search circle radius in pixels around the foot center. Defaults to 24px.

        Returns:
            Overlap ratio between 0.0 (strictly off-road / sidewalk) and 1.0 (fully on roadway).
        """
        h, w = road_mask.shape[:2]
        cx = int(np.clip(foot_x_norm * w, 0, w - 1))
        cy = int(np.clip(foot_y_norm * h, 0, h - 1))

        y_min = max(0, cy - radius_px)
        y_max = min(h, cy + radius_px + 1)
        x_min = max(0, cx - radius_px)
        x_max = min(w, cx + radius_px + 1)

        patch = road_mask[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return 0.0

        y_indices, x_indices = np.ogrid[y_min - cy:y_max - cy, x_min - cx:x_max - cx]
        dist_from_center = np.sqrt(x_indices**2 + y_indices**2)
        circle_mask = dist_from_center <= radius_px

        if not np.any(circle_mask):
            return 0.0

        overlap = np.mean(patch[circle_mask])
        return float(overlap)
