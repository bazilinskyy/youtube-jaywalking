"""
End-to-End Production Jaywalking Detection Pipeline (Exp57/Exp58 Architecture).
"""

import time
from typing import Any, Dict

from src.perception.vlm_classifier import VLMClassifier, CANONICAL_CLASSIFICATION_PROMPT
from src.perception.pedestrian_tracking import PedestrianTracker
from src.perception.road_segmentation import RoadSegmenter
from src.pipeline.frame_sampler import FrameSampler
from src.pipeline.context_router import ContextRouter
from src.pipeline.decision_engine import DecisionEngine
from src.utils.video_utils import encode_frame_to_base64


class JaywalkingPipeline:
    """
    Unified end-to-end jaywalking detection system combining pose tracking,
    road segmentation, multi-temporal VLM classification, and wide context verification.
    """

    def __init__(
        self,
        vlm_model: str = "qwen2.5vl:7b",
        pose_model_path: str = "yolo26x-pose.pt",
        seg_model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
        device: str = "cuda",
    ):
        self.vlm = VLMClassifier(model_name=vlm_model)
        self.tracker = PedestrianTracker(model_path=pose_model_path)
        self.segmenter = RoadSegmenter(model_name=seg_model_name, device=device)
        self.sampler = FrameSampler(num_keyframes=3)
        self.router = ContextRouter(self.vlm)
        self.engine = DecisionEngine()

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Processes an input video clip and returns the prediction and diagnostic metadata.
        """
        t0 = time.time()

        # 1. Sample 3 Keyframes across video
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
