"""Video decoding, frame sampling, and base64 encoding utilities.

This module provides helper functions to decode video streams, extract equidistant frames,
and convert OpenCV BGR images into base64 JPEG strings for vision-language model querying.
"""

import base64
from typing import List, Tuple
import cv2
import numpy as np


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encodes a BGR OpenCV image into a JPEG base64 string.

    Args:
        frame: OpenCV image in BGR format (H, W, 3).
        quality: JPEG compression quality from 1 to 100. Defaults to 85.

    Returns:
        UTF-8 encoded base64 string representing the compressed JPEG image.

    Raises:
        ValueError: If JPEG compression fails.
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise ValueError("Failed to encode frame to JPEG format.")
    return base64.b64encode(buffer).decode("utf-8")


def extract_equidistant_frames(
    vpath: str,
    num_frames: int = 3,
) -> Tuple[List[np.ndarray], List[int], float, int]:
    """Extracts equidistant frames across the video duration.

    Args:
        vpath: Path to the video file (.mp4, .avi).
        num_frames: Number of equidistant frames to extract. Defaults to 3.

    Returns:
        A tuple of (frames, indices, fps, total_frames):
            - frames: List of decoded BGR image arrays.
            - indices: List of 0-based frame indices sampled.
            - fps: Video frame rate.
            - total_frames: Total number of frames in the video.

    Raises:
        FileNotFoundError: If the video cannot be opened.
        ValueError: If the video file contains 0 frames.
    """
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at: {vpath}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if tot_frames <= 0:
        cap.release()
        raise ValueError(f"Video {vpath} contains 0 frames.")

    if num_frames == 3:
        indices = [0, max(0, tot_frames // 2), max(0, tot_frames - 1)]
    else:
        indices = [int(x) for x in np.linspace(0, tot_frames - 1, num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
        else:
            # Fallback black frame if frame decoding fails
            frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))

    cap.release()
    return frames, indices, fps, tot_frames
