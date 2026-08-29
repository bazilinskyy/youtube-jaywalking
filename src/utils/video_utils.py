"""
Video decoding, frame sampling, and base64 encoding utilities.
"""

import base64
from typing import List, Optional, Tuple
import cv2
import numpy as np


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encodes a BGR OpenCV image into a JPEG base64 string."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise ValueError("Failed to encode frame to JPEG format.")
    return base64.b64encode(buffer).decode("utf-8")


def extract_equidistant_frames(
    vpath: str,
    num_frames: int = 3,
) -> Tuple[List[np.ndarray], List[int], float, int]:
    """
    Extracts N equidistant frames across the video lifespan.
    Returns: (frames, indices, fps, total_frames)
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
            # Fallback black frame if corrupt
            frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))
            
    cap.release()
    return frames, indices, fps, tot_frames
