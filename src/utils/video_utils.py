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
    # Configure OpenCV JPEG compression quality level
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # Compress raw pixel array into in-memory JPEG byte buffer
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    # Check for compression failure or invalid image buffer
    if not success:
        raise ValueError("Failed to encode frame to JPEG format.")
    # Convert binary buffer into ASCII base64 string
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
    # Initialize video capture stream
    cap = cv2.VideoCapture(vpath)
    # Validate stream accessibility
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at: {vpath}")

    # Read video frame rate (fallback to 30.0 if not reported)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Read total frame count
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Guard against empty/corrupted video files
    if tot_frames <= 0:
        cap.release()
        raise ValueError(f"Video {vpath} contains 0 frames.")

    # Calculate exact equidistant sampling indices
    if num_frames == 3:
        # Canonical 3-frame layout: start (0), midpoint (50%), and endpoint (100%)
        indices = [0, max(0, tot_frames // 2), max(0, tot_frames - 1)]
    else:
        # Linear space partition for N frames
        indices = [int(x) for x in np.linspace(0, tot_frames - 1, num_frames)]

    frames = []
    # Seek and extract each calculated keyframe index
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
        else:
            # Fallback zero-filled frame if specific frame decoding fails
            frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))

    # Clean up video capture resources
    cap.release()
    return frames, indices, fps, tot_frames
