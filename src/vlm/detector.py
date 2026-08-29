import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import get_vlm_config
from src.vlm.client import OllamaClient, encode_frame_to_base64
from src.vlm.prompts import get_prompt


class VLMJaywalkingDetector:
    """End-to-end VLM detector for video crossing compliance and jaywalking."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        prompt_name: str = "canonical",
        custom_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        temperature: Optional[float] = None,
        use_boundary_context: bool = False,
        use_pedestrian_motion: bool = False,
        use_vehicle_context: bool = False,
        temporal_mode: Optional[bool] = None,
        min_votes_for_jaywalking: int = 2,
    ) -> None:
        cfg = get_vlm_config()
        self.model_name = model_name or cfg.get("model", "qwen2.5vl:7b")
        self.ollama_url = ollama_url or cfg.get("ollama_url", "http://localhost:11434/api/chat")
        self.num_frames = num_frames or cfg.get("num_frames", 3)
        self.temperature = temperature if temperature is not None else cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 10)
        self.jpeg_quality = cfg.get("jpeg_quality", 85)
        self.timeout_seconds = cfg.get("timeout_seconds", 60)
        self.seed = cfg.get("seed", 42)
        self.use_boundary_context = use_boundary_context
        self.use_pedestrian_motion = use_pedestrian_motion or (
            prompt_name in ("temporal_motion", "temporal_vehicle_motion", "v4b"))
        self.use_vehicle_context = use_vehicle_context or (prompt_name in ("temporal_vehicle_motion", "v4b"))
        self.temporal_mode = temporal_mode if temporal_mode is not None else (
            prompt_name in ("temporal", "temporal_motion", "temporal_vehicle_motion", "v4b"))
        self.min_votes_for_jaywalking = min_votes_for_jaywalking

        self.prompt = custom_prompt or get_prompt(prompt_name)
        self.client = OllamaClient(
            base_url=self.ollama_url,
            model=self.model_name,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )

        self._boundary_detector = None
        self._yolo = None
        if self.use_boundary_context:
            from src.cv.boundary import BoundaryDetector
            from ultralytics import YOLO
            self._boundary_detector = BoundaryDetector()
            self._yolo = YOLO("models/yolo11x.pt")

        self._motion_extractor = None
        if self.use_pedestrian_motion:
            from src.cv.pedestrian_motion import PedestrianMotionExtractor
            self._motion_extractor = PedestrianMotionExtractor()

        self._vehicle_extractor = None
        if self.use_vehicle_context:
            from src.cv.vehicle_state import VehicleStateExtractor
            self._vehicle_extractor = VehicleStateExtractor()

    def sample_keyframes(
        self, video_path: Union[str, Path], num_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Extracts equidistant keyframes across the duration of a video clip."""
        path_str = str(video_path)
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        k = num_frames or self.num_frames

        if total_frames <= 0:
            cap.release()
            return [], []

        if total_frames <= k:
            indices = list(range(total_frames))
        else:
            indices = [int(i * total_frames / k) for i in range(k)]

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

    def parse_response(self, raw_text: str) -> str:
        """Parses the raw text output into canonical label."""
        upper = raw_text.strip().upper()
        if "JAYWALKING" in upper:
            return "jaywalking"
        if "COMPLIANT" in upper:
            return "compliant"
        return "unknown"

    def classify_frame(self, frame: np.ndarray, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Classifies a single video frame with the VLM."""
        p = prompt or self.prompt
        b64 = encode_frame_to_base64(frame, quality=self.jpeg_quality)
        raw = self.client.generate_chat(prompt=p, base64_images=b64)
        label = self.parse_response(raw)
        return {
            "prediction": label,
            "raw_response": raw,
        }

    def predict(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Runs VLM jaywalking detection on a video clip with majority voting."""
        t0 = time.time()
        frames, indices = self.sample_keyframes(video_path)
        if not frames:
            return {
                "prediction": "unknown",
                "confidence": "low",
                "reason": "Failed to read frames from video",
                "frame_votes": [],
                "raw_responses": [],
                "frame_indices": [],
                "elapsed_seconds": round(time.time() - t0, 3),
            }

        if self.temporal_mode:
            b64_list = [encode_frame_to_base64(f, quality=self.jpeg_quality) for f in frames]
            motion_info = None
            vehicle_info = None
            prompt_to_use = self.prompt

            if self.use_pedestrian_motion and self._motion_extractor is not None:
                motion_info = self._motion_extractor.extract(video_path)
                if "{pedestrian_motion}" in prompt_to_use:
                    prompt_to_use = prompt_to_use.replace("{pedestrian_motion}", motion_info["formatted_context"])
                else:
                    prompt_to_use = f"{prompt_to_use}\n\n{motion_info['formatted_context']}"

            if self.use_vehicle_context and self._vehicle_extractor is not None:
                vehicle_info = self._vehicle_extractor.extract(video_path)
                if "{vehicle_interaction}" in prompt_to_use:
                    prompt_to_use = prompt_to_use.replace("{vehicle_interaction}", vehicle_info["formatted_context"])
                else:
                    prompt_to_use = f"{prompt_to_use}\n\n{vehicle_info['formatted_context']}"

            raw = self.client.generate_chat(prompt=prompt_to_use, base64_images=b64_list)
            label = self.parse_response(raw)
            elapsed = round(time.time() - t0, 3)
            result = {
                "prediction": label,
                "confidence": "high" if label in ("jaywalking", "compliant") else "low",
                "reason": f"Temporal multi-frame reasoning ({label})",
                "frame_votes": [label],
                "raw_responses": [raw],
                "frame_indices": indices,
                "elapsed_seconds": elapsed,
            }
            if motion_info is not None:
                result["pedestrian_motion"] = motion_info
            if vehicle_info is not None:
                result["vehicle_interaction"] = vehicle_info
            return result

        boundary_info = None
        if self.use_boundary_context and self._boundary_detector is not None:
            boundary_info = self._boundary_detector.detect(video_path)

        votes: List[str] = []
        raw_responses: List[str] = []
        spatial_positions: List[str] = []
        for f in frames:
            if boundary_info is not None:
                from src.cv.boundary import get_pedestrian_spatial_position
                pos = get_pedestrian_spatial_position(f, boundary_info, self._yolo)
                spatial_positions.append(pos)
                frame_prompt = f"{self.prompt}\n\nSpatial context:\npedestrian_position = {pos}"
                res = self.classify_frame(f, prompt=frame_prompt)
            else:
                res = self.classify_frame(f)
            votes.append(res["prediction"])
            raw_responses.append(res["raw_response"])

        counts = Counter(votes)
        jw_votes = counts.get("jaywalking", 0)
        comp_votes = counts.get("compliant", 0)

        if jw_votes >= self.min_votes_for_jaywalking:
            final_pred = "jaywalking"
            confidence = "high" if jw_votes == len(votes) else "medium"
        else:
            final_pred = "compliant"
            confidence = "high" if comp_votes == len(votes) else "medium"

        elapsed = round(time.time() - t0, 3)
        res_dict = {
            "prediction": final_pred,
            "confidence": confidence,
            "reason": (
                f"Vote margin ({jw_votes} jaywalking vs {comp_votes} compliant, "
                f"min_req={self.min_votes_for_jaywalking})"
            ),
            "frame_votes": votes,
            "raw_responses": raw_responses,
            "frame_indices": indices,
            "elapsed_seconds": elapsed,
        }

        if spatial_positions:
            res_dict["spatial_positions"] = spatial_positions
        return res_dict
