"""Temporal frame extraction and keyframe selection strategies.

This module provides the FrameSampler class, which handles extracting equidistant keyframes
and multi-timestamp frames across the video lifespan for visual reasoning and segmentation.
"""

from typing import List, Tuple
import cv2
import numpy as np
from src.utils.video_utils import extract_equidistant_frames


class FrameSampler:
    """Extracts multi-temporal observation frames for classification and context analysis."""

    def __init__(self, num_keyframes: int = 3) -> None:
        """Initializes the frame sampler.

        Args:
            num_keyframes: Number of equidistant keyframes to sample. Defaults to 3.
        """
        self.num_keyframes = num_keyframes

    def sample_keyframes(self, video_path: str) -> Tuple[List[np.ndarray], List[int], float, int]:
        """Samples equidistant keyframes across the video duration.

        Args:
            video_path: Path to the input video file.

        Returns:
            A tuple of (frames, indices, fps, total_frames):
                - frames: List of sampled BGR images.
                - indices: List of 0-based frame indices.
                - fps: Video frame rate.
                - total_frames: Total number of frames in the video.
        """
        return extract_equidistant_frames(video_path, num_frames=self.num_keyframes)

    def sample_temporal_timestamps(
        self,
        video_path: str,
        fractions: List[float] = [0.25, 0.50, 0.75]
    ) -> List[np.ndarray]:
        """Samples specific fractional timestamp frames across the video lifespan.

        Args:
            video_path: Path to the input video file.
            fractions: List of fractional progress points (0.0 to 1.0) to sample. Defaults to [0.25, 0.50, 0.75].

        Returns:
            List of decoded OpenCV BGR frames corresponding to the requested fractional timestamps.
        """
        cap = cv2.VideoCapture(video_path)
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for frac in fractions:
            idx = int(tot_frames * frac)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            else:
                frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))

        cap.release()
        return frames
