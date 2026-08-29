import unittest
from pathlib import Path
import numpy as np

from src.config import get_vlm_config, get_cv_config
from src.data_loader import load_ground_truth_records
from src.vlm.prompts import get_prompt, CANONICAL_PROMPT
from src.vlm.detector import VLMJaywalkingDetector
from src.pipeline import get_pipeline
from evaluation.metrics import compute_metrics


class TestJaywalkingPipeline(unittest.TestCase):

    def test_config_loader(self):
        vlm_cfg = get_vlm_config()
        self.assertEqual(vlm_cfg.get("model"), "qwen2.5vl:7b")
        self.assertEqual(vlm_cfg.get("num_frames"), 3)

        cv_cfg = get_cv_config()
        self.assertTrue(Path(cv_cfg["yolo_model"]).exists())

    def test_ground_truth_loader(self):
        records = load_ground_truth_records(only_evaluable=True)
        self.assertEqual(len(records), 39)
        for r in records:
            self.assertIn(r["ground_truth"], ("jaywalking", "compliant"))
            self.assertTrue(Path(r["video_path"]).exists(), f"Video missing: {r['video_path']}")

    def test_prompt_presets(self):
        prompt = get_prompt("canonical")
        self.assertEqual(prompt, CANONICAL_PROMPT)
        self.assertIn("JAYWALKING or COMPLIANT", prompt)

    def test_metrics_calculation(self):
        dummy_results = [
            {"ground_truth": "jaywalking", "prediction": "jaywalking"},
            {"ground_truth": "jaywalking", "prediction": "jaywalking"},
            {"ground_truth": "compliant", "prediction": "compliant"},
            {"ground_truth": "compliant", "prediction": "jaywalking"},  # FP
        ]
        m = compute_metrics(dummy_results)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["tn"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 0)
        self.assertEqual(m["accuracy"], 75.0)
        self.assertEqual(m["recall"], 100.0)

    def test_all_ground_truth_records(self):
        all_records = load_ground_truth_records(only_evaluable=False)
        self.assertEqual(len(all_records), 50)
        evaluable = [r for r in all_records if r["is_evaluated"]]
        non_evaluable = [r for r in all_records if not r["is_evaluated"]]
        self.assertEqual(len(evaluable), 39)
        self.assertEqual(len(non_evaluable), 11)

    def test_vlm_response_parser(self):
        detector = VLMJaywalkingDetector()
        self.assertEqual(detector.parse_response("JAYWALKING"), "jaywalking")
        self.assertEqual(detector.parse_response("The classification is COMPLIANT."), "compliant")
        self.assertEqual(detector.parse_response("Jaywalking - crossing on red"), "jaywalking")
        self.assertEqual(detector.parse_response("UNCERTAIN"), "unknown")

    def test_vlm_keyframe_sampling(self):
        detector = VLMJaywalkingDetector()
        test_video = Path(__file__).resolve().parents[1] / "data" / "raw_clips" / "video_0014.mp4"
        if test_video.exists():
            frames, indices = detector.sample_keyframes(test_video, num_frames=3)
            self.assertEqual(len(frames), 3)
            self.assertEqual(len(indices), 3)
            self.assertIsInstance(frames[0], np.ndarray)

    def test_pipeline_modes(self):
        p_balanced = get_pipeline(mode="balanced")
        self.assertEqual(p_balanced.min_votes_for_jaywalking, 2)

        p_hp = get_pipeline(mode="high_precision")
        self.assertEqual(p_hp.min_votes_for_jaywalking, 3)

        p_safety = get_pipeline(mode="safety")
        self.assertEqual(p_safety.min_votes_for_jaywalking, 1)

        p_custom = get_pipeline(mode="vlm", min_votes=3)
        self.assertEqual(p_custom.min_votes_for_jaywalking, 3)


if __name__ == "__main__":
    unittest.main()
