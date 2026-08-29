import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import get_vlm_config
from src.vlm.client import OllamaClient, encode_frame_to_base64


class FullVideoVLMDetector:
    """Full-video sequence VLM baseline detector using Chain-of-Causation (CoC) reasoning (qwen2.5vl:7b).

    Processes multi-frame video sequences using 5-step Chain-of-Causation reasoning.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_frames: int = 5,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        cfg = get_vlm_config()
        self.model_name = model_name or cfg.get("model", "qwen2.5vl:7b")
        self.max_frames = max_frames
        self.temperature = temperature
        self.seed = seed

        self.client = OllamaClient(
            model=self.model_name,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=300,
        )

        self.coc_prompt = (
            "Analyze the full video sequence of pedestrian and vehicle interactions.\n"
            "Produce Chain-of-Causation (CoC) reasoning steps:\n"
            "1. Pedestrian Trajectory & Location: [sidewalk / curb / roadway]\n"
            "2. Infrastructure & Right-of-Way: [marked crosswalk / signal / none]\n"
            "3. Vehicle Kinematic Response: [yielding / decelerating / accelerating / none]\n"
            "4. Causal Analysis: Explain why the crossing is legal compliance or an illegal violation.\n"
            "5. Final Classification: Output EXACTLY either JAYWALKING or COMPLIANT."
        )

    def extract_full_video_frames(
        self, video_path: Union[str, Path], target_fps: int = 5
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Extracts full video frame sequence sampled across clip duration."""
        path_str = str(video_path)
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return [], []

        indices = np.linspace(0, total_frames - 1, num=self.max_frames, dtype=int).tolist()

        frames: List[np.ndarray] = []
        frame_indices: List[int] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
                frame_indices.append(idx)

        cap.release()
        return frames, frame_indices

    def parse_coc_response(self, raw_text: str) -> Dict[str, str]:
        """Parses Chain-of-Causation response text into verdict and reasoning components."""
        upper = raw_text.upper()
        if "JAYWALKING" in upper and "COMPLIANT" not in upper:
            prediction = "jaywalking"
        elif "COMPLIANT" in upper and "JAYWALKING" not in upper:
            prediction = "compliant"
        else:
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            last_line = lines[-1].upper() if lines else ""
            if "JAYWALKING" in last_line:
                prediction = "jaywalking"
            elif "COMPLIANT" in last_line:
                prediction = "compliant"
            else:
                prediction = "jaywalking" if "JAYWALKING" in upper else "compliant"

        return {
            "prediction": prediction,
            "chain_of_causation": raw_text,
        }

    def extract_segment_frames(
        self,
        video_path: Union[str, Path],
        start_frame: int,
        end_frame: int,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Extracts 5 frames sampled within a specific segment [start_frame, end_frame]."""
        path_str = str(video_path)
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            return [], []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return [], []

        s_idx = max(0, min(start_frame, total_frames - 1))
        e_idx = max(s_idx, min(end_frame, total_frames - 1))

        if e_idx == s_idx:
            indices = [s_idx] * self.max_frames
        else:
            raw_indices = np.linspace(s_idx, e_idx, num=self.max_frames, dtype=int)
            indices = [int(x) for x in raw_indices]

        frames: List[np.ndarray] = []
        frame_indices: List[int] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
                frame_indices.append(idx)

        cap.release()
        return frames, frame_indices

    def predict(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Runs full-video sequence Chain-of-Causation (CoC) inference on a video clip."""
        t0 = time.time()
        frames, indices = self.extract_full_video_frames(video_path)

        if not frames:
            return {
                "prediction": "unknown",
                "confidence": "low",
                "reason": "Failed to extract frames from video",
                "elapsed_seconds": round(time.time() - t0, 3),
            }

        b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
        raw_response = self.client.generate_chat(prompt=self.coc_prompt, base64_images=b64_list)

        parsed = self.parse_coc_response(raw_response)
        elapsed = round(time.time() - t0, 3)

        return {
            "prediction": parsed["prediction"],
            "confidence": "high",
            "reason": f"Full-video CoC reasoning ({len(frames)} frames)",
            "chain_of_causation": parsed["chain_of_causation"],
            "num_video_frames": len(frames),
            "frame_indices": indices,
            "elapsed_seconds": elapsed,
        }


class EventLocalizedVLMDetector(FullVideoVLMDetector):
    """Event-Localized VLM Baseline Detector

    Uses ByteTrack / PedestrianMotionExtractor to detect candidate pedestrian crossing events,
    localizes 5 frames specifically within the crossing segment interval [start_frame, end_frame],
    and runs Chain-of-Causation reasoning through the VLM baseline detector.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_frames: int = 5,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_frames=max_frames,
            temperature=temperature,
            seed=seed,
        )
        from src.cv.pedestrian_motion import PedestrianMotionExtractor
        self.motion_extractor = PedestrianMotionExtractor()

    def predict(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        t0 = time.time()

        # Step 1 & 2: Detect candidate crossing event segment start and end frames
        s_frame, e_frame, motion_info = self.motion_extractor.detect_crossing_segment(video_path)

        # Step 3: Sample 5 frames within the localized crossing segment interval
        if s_frame is not None and e_frame is not None:
            frames, indices = self.extract_segment_frames(video_path, s_frame, e_frame)
            reason_msg = f"Event-localized CoC reasoning (segment frames {s_frame}-{e_frame})"
        else:
            frames, indices = self.extract_full_video_frames(video_path)
            reason_msg = "Full-video CoC reasoning fallback (no localized track)"

        if not frames:
            return {
                "prediction": "unknown",
                "confidence": "low",
                "reason": "Failed to extract segment frames from video",
                "elapsed_seconds": round(time.time() - t0, 3),
            }

        # Step 4: Send 5 segment frames through VLM baseline detector & CoC prompt
        b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
        raw_response = self.client.generate_chat(prompt=self.coc_prompt, base64_images=b64_list)

        # Step 5: Produce final event prediction
        parsed = self.parse_coc_response(raw_response)
        elapsed = round(time.time() - t0, 3)

        return {
            "prediction": parsed["prediction"],
            "confidence": "high",
            "reason": reason_msg,
            "chain_of_causation": parsed["chain_of_causation"],
            "num_video_frames": len(frames),
            "frame_indices": indices,
            "crossing_segment": [s_frame, e_frame] if s_frame is not None else None,
            "motion_info": motion_info,
            "elapsed_seconds": elapsed,
        }


# Aliases for backward compatibility
AlpamayoFullVideoDetector = FullVideoVLMDetector
EventLocalizedAlpamayoDetector = EventLocalizedVLMDetector
