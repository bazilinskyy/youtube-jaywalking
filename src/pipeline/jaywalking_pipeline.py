"""End-to-End Production Jaywalking Detection Pipeline (Exp57/Exp58 Architecture).

This module defines the JaywalkingPipeline class, which orchestrates keyframe extraction,
zero-shot VLM consensus classification, kinematic pedestrian pose tracking, semantic road
segmentation, wide-scene contextual verification, and rule-based decision synthesis.
"""

import time
from typing import Any, Dict

from src.perception.pedestrian_tracking import PedestrianTracker
from src.perception.road_segmentation import RoadSegmenter
from src.perception.vlm_classifier import CANONICAL_CLASSIFICATION_PROMPT, VLMClassifier
from src.pipeline.context_router import ContextRouter
from src.pipeline.decision_engine import DecisionEngine
from src.pipeline.frame_sampler import FrameSampler
from src.utils.video_utils import encode_frame_to_base64


class JaywalkingPipeline:
    """Unified end-to-end jaywalking detection system."""

    def __init__(
        self,
        vlm_model: str = "qwen2.5vl:7b",
        pose_model_path: str = "yolo26x-pose.pt",
        seg_model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
        device: str = "cuda",
    ) -> None:
        """Initializes all perception, tracking, segmentation, and decision submodules.

        Args:
            vlm_model: Identifier for the local Ollama VLM model.
            pose_model_path: Path to YOLO26x-Pose model weights.
            seg_model_name: HuggingFace model path for SegFormer-B0 Cityscapes segmentation.
            device: Target execution device ('cuda' or 'cpu').
        """
        self.vlm = VLMClassifier(model_name=vlm_model)
        self.tracker = PedestrianTracker(model_path=pose_model_path)
        self.segmenter = RoadSegmenter(model_name=seg_model_name, device=device)
        self.sampler = FrameSampler(num_keyframes=3)
        self.router = ContextRouter(self.vlm)
        self.engine = DecisionEngine()

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Processes an input video clip and returns the prediction and diagnostic metadata.

        Executes the 6-stage multimodal inference sequence:
            1. Extracts 3 chronological keyframes (0%, 50%, 100%).
            2. Computes independent VLM classification votes on each keyframe.
            3. Tracks pedestrian pose and extracts lateral displacement, base coordinate, and duration.
            4. Evaluates multi-temporal road surface segmentation and foot-road overlap.
            5. Gated wide-scene context verification for crosswalk, public road, and junction status.
            6. Synthesizes final classification decision and reasoning path via DecisionEngine.

        Args:
            video_path: Filepath to the input video file (.mp4).

        Returns:
            Dictionary containing:
                - prediction (str): Final label ('JAYWALKING' or 'COMPLIANT').
                - decision_path (str): Human-readable explanation of the triggering decision rule.
                - votes (List[str]): List of 3 independent VLM frame votes.
                - lateral_displacement (float): Maximum transverse displacement across track.
                - mean_y (float): Average vertical coordinate of pedestrian track base.
                - track_duration_sec (float): Duration in seconds of the primary pedestrian track.
                - road_overlap (float): Ratio of drivable road pixels at pedestrian foot base.
                - crosswalk_status (str): Context verification result for marked crosswalks.
                - road_structure_status (str): Context verification result for roadway structure.
                - junction_status (str): Context verification result for intersection junctions.
                - latency_sec (float): Total processing time in seconds.
        """
        t0 = time.time()

        # 1. Sample 3 Keyframes across video (Start: 0%, Mid: 50%, End: 100%)
        frames, indices, fps, tot_frames = self.sampler.sample_keyframes(video_path)
        mid_frame = frames[1] if len(frames) > 1 else frames[0]

        # 2. VLM 3-Frame Unanimous Vote
        votes = []
        for fr in frames:
            b64 = encode_frame_to_base64(fr, quality=85)
            resp = self.vlm.query(CANONICAL_CLASSIFICATION_PROMPT, b64)
            vote = "JAYWALKING" if "JAYWALKING" in resp.upper() else "COMPLIANT"
            votes.append(vote)

        p_unanimous = "JAYWALKING" if votes.count("JAYWALKING") == 3 else "COMPLIANT"

        # 3. Pedestrian Trajectory Tracking
        lat_disp, mean_y, track_dur, _ = self.tracker.track_video(video_path, fps=fps)

        # 4. Multi-temporal Road Surface Segmentation
        temporal_frames = self.sampler.sample_temporal_timestamps(video_path, fractions=[0.25, 0.50, 0.75])
        ov_samples = []
        for fr in temporal_frames:
            rmask = self.segmenter.segment_road_mask(fr)
            ov = self.segmenter.evaluate_foot_road_overlap(rmask, 0.50, mean_y, radius_px=24)
            ov_samples.append(ov)

        static_road_ov = ov_samples[1] if len(ov_samples) > 1 else 0.0

        # 5. Context Verification (Only executed if base pipeline indicates crossing candidate)
        resp_cw = "NO_CROSSWALK"
        resp_road = "PUBLIC_STREET"
        resp_junc = "UNREGULATED_MIDBLOCK"

        if p_unanimous == "JAYWALKING" or (votes.count("JAYWALKING") == 2 and track_dur <= 1.5):
            resp_cw, resp_road, resp_junc = self.router.verify_scene_context(mid_frame)

        # 6. Final Decision Synthesis
        prediction, reason = self.engine.evaluate(
            votes=votes,
            lateral_displacement=lat_disp,
            mean_y=mean_y,
            track_duration_sec=track_dur,
            static_road_overlap=static_road_ov,
            crosswalk_status=resp_cw,
            road_structure_status=resp_road,
            junction_status=resp_junc,
        )

        elapsed = round(time.time() - t0, 2)

        return {
            "prediction": prediction,
            "decision_path": reason,
            "votes": votes,
            "lateral_displacement": lat_disp,
            "mean_y": mean_y,
            "track_duration_sec": track_dur,
            "road_overlap": round(static_road_ov, 3),
            "crosswalk_status": resp_cw,
            "road_structure_status": resp_road,
            "junction_status": resp_junc,
            "latency_sec": elapsed,
        }
