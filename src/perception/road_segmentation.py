"""
Semantic road surface segmentation using SegFormer-B0.
"""

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class RoadSegmenter:
    """
    SegFormer-B0 Cityscapes segmentation model for identifying drivable roadway pixels.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def segment_road_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Segments the road mask from an image.
        Returns a binary uint8 mask where 1 indicates Road (Cityscapes class 0).
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
        # Cityscapes: Class 0 is 'road'
        road_mask = (pred_seg == 0).astype(np.uint8)
        return road_mask

    def evaluate_foot_road_overlap(
        self,
        road_mask: np.ndarray,
        foot_x_norm: float,
        foot_y_norm: float,
        radius_px: int = 24,
    ) -> float:
        """
        Calculates the ratio of road pixels in a circular neighborhood around the pedestrian's base coordinates.
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
