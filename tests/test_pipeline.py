"""Unit tests for the production Jaywalking Detection architecture (Exp57/Exp58)."""

import unittest
import numpy as np

from src.pipeline.decision_engine import DecisionEngine
from src.perception.vlm_classifier import (
    CANONICAL_CLASSIFICATION_PROMPT,
    CROSSWALK_VERIFIER_PROMPT,
    PUBLIC_ROADWAY_VERIFIER_PROMPT,
    LEGAL_JUNCTION_VERIFIER_PROMPT,
)
from src.utils.metrics import calculate_classification_metrics
from src.utils.video_utils import encode_frame_to_base64
import common


class TestProductionPipeline(unittest.TestCase):
    """Verifies core production pipeline components and decision logic."""

    def setUp(self) -> None:
        """Initializes decision engine and test fixtures."""
        self.engine = DecisionEngine()

    def test_config_loader(self) -> None:
        """Tests that default configuration keys are properly loaded."""
        import json
        import os
        with open(os.path.join(common.root_dir, "default.config")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["vlm_model"], "qwen2.5vl:7b")
        self.assertEqual(cfg["vlm_num_keyframes"], 3)
        self.assertEqual(cfg["tracking_model_path"], "yolo26x-pose.pt")

    def test_decision_engine_rules(self) -> None:
        """Tests each decision path in the frozen Exp57 decision engine."""
        # 1. Unanimous Jaywalking -> Public Street = JAYWALKING
        pred, reason = self.engine.evaluate(
            votes=["JAYWALKING", "JAYWALKING", "JAYWALKING"],
            lateral_displacement=0.5,
            mean_y=0.6,
            track_duration_sec=3.0,
            static_road_overlap=0.8,
            crosswalk_status="NO_CROSSWALK",
            road_structure_status="PUBLIC_STREET",
            junction_status="UNREGULATED_MIDBLOCK",
        )
        self.assertEqual(pred, "JAYWALKING")
        self.assertIn("unanimous VLM", reason)

        # 2. Unanimous Jaywalking + Legal Crosswalk = COMPLIANT
        pred, reason = self.engine.evaluate(
            votes=["JAYWALKING", "JAYWALKING", "JAYWALKING"],
            lateral_displacement=0.5,
            mean_y=0.6,
            track_duration_sec=3.0,
            static_road_overlap=0.8,
            crosswalk_status="LEGAL_CROSSWALK",
            road_structure_status="PUBLIC_STREET",
            junction_status="UNREGULATED_MIDBLOCK",
        )
        self.assertEqual(pred, "COMPLIANT")
        self.assertIn("crosswalk", reason.lower())

        # 3. Unanimous Jaywalking + Driveway Apron Filter = COMPLIANT
        pred, reason = self.engine.evaluate(
            votes=["JAYWALKING", "JAYWALKING", "JAYWALKING"],
            lateral_displacement=0.1,
            mean_y=0.90,
            track_duration_sec=7.0,
            static_road_overlap=0.10,
            crosswalk_status="NO_CROSSWALK",
            road_structure_status="PUBLIC_STREET",
            junction_status="UNREGULATED_MIDBLOCK",
        )
        self.assertEqual(pred, "COMPLIANT")
        self.assertIn("bumper", reason.lower())

        # 4. 2/3 Fast-Crossing Sprint Dash = JAYWALKING
        pred, reason = self.engine.evaluate(
            votes=["JAYWALKING", "JAYWALKING", "COMPLIANT"],
            lateral_displacement=0.25,
            mean_y=0.6,
            track_duration_sec=1.2,
            static_road_overlap=0.5,
            crosswalk_status="NO_CROSSWALK",
            road_structure_status="PUBLIC_STREET",
            junction_status="UNREGULATED_MIDBLOCK",
        )
        self.assertEqual(pred, "JAYWALKING")
        self.assertIn("dash", reason.lower())

        # 5. Compliant Consensus = COMPLIANT
        pred, reason = self.engine.evaluate(
            votes=["COMPLIANT", "COMPLIANT", "COMPLIANT"],
            lateral_displacement=0.05,
            mean_y=0.5,
            track_duration_sec=4.0,
            static_road_overlap=0.0,
            crosswalk_status="NO_CROSSWALK",
            road_structure_status="PUBLIC_STREET",
            junction_status="UNREGULATED_MIDBLOCK",
        )
        self.assertEqual(pred, "COMPLIANT")
        self.assertIn("Compliant consensus", reason)

    def test_metrics_calculation(self) -> None:
        """Tests standard classification metrics computation."""
        y_true = ["JAYWALKING", "JAYWALKING", "COMPLIANT", "COMPLIANT"]
        y_pred = ["JAYWALKING", "JAYWALKING", "COMPLIANT", "JAYWALKING"]  # 1 FP

        m = calculate_classification_metrics(y_true, y_pred)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["tn"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 0)
        self.assertEqual(m["accuracy"], 75.0)
        self.assertEqual(m["recall"], 100.0)

    def test_prompt_constants(self) -> None:
        """Verifies core VLM prompt text definitions."""
        self.assertIn("JAYWALKING or COMPLIANT", CANONICAL_CLASSIFICATION_PROMPT)
        self.assertIn("LEGAL_CROSSWALK", CROSSWALK_VERIFIER_PROMPT)
        self.assertIn("PUBLIC_STREET", PUBLIC_ROADWAY_VERIFIER_PROMPT)
        self.assertIn("LEGAL_JUNCTION_CROSSING", LEGAL_JUNCTION_VERIFIER_PROMPT)

    def test_frame_encoding(self) -> None:
        """Verifies base64 frame encoding utility."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        b64 = encode_frame_to_base64(dummy_frame)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 0)


if __name__ == "__main__":
    unittest.main()
