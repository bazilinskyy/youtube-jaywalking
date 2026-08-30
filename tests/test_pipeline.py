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

    def test_pedestrian_tracker_mean_x(self) -> None:
        """Verifies PedestrianTracker computes mean_x from tracked pedestrian detections."""
        from unittest.mock import MagicMock
        import torch
        from src.perception.pedestrian_tracking import PedestrianTracker

        tracker = PedestrianTracker.__new__(PedestrianTracker)
        tracker.tracker_config = "configs/botsort_custom.yaml"
        tracker.conf = 0.25
        tracker.iou = 0.5

        # Create mock tracking results with an off-center pedestrian (cx ~ 0.20)
        # xywhn format: [cx, cy, w, h]
        mock_res1 = MagicMock()
        mock_res1.boxes.id = torch.tensor([1])
        mock_res1.boxes.xywhn = torch.tensor([[0.18, 0.60, 0.08, 0.20]])

        mock_res2 = MagicMock()
        mock_res2.boxes.id = torch.tensor([1])
        mock_res2.boxes.xywhn = torch.tensor([[0.20, 0.62, 0.08, 0.20]])

        mock_res3 = MagicMock()
        mock_res3.boxes.id = torch.tensor([1])
        mock_res3.boxes.xywhn = torch.tensor([[0.22, 0.64, 0.08, 0.20]])

        tracker.model = MagicMock()
        tracker.model.track.return_value = [mock_res1, mock_res2, mock_res3]

        lat_disp, mean_x, mean_y, track_dur, dom_frames = tracker.track_video("mock_video.mp4", fps=30.0)

        # Expected: mean_x = mean([0.18, 0.20, 0.22]) = 0.20
        self.assertAlmostEqual(mean_x, 0.20, places=2)
        # Expected: lat_disp = abs(0.22 - 0.18) = 0.04
        self.assertAlmostEqual(lat_disp, 0.04, places=2)
        # Expected: mean_y = mean([0.60 + 0.10, 0.62 + 0.10, 0.64 + 0.10]) = 0.72
        self.assertAlmostEqual(mean_y, 0.72, places=2)
        self.assertEqual(len(dom_frames), 3)


if __name__ == "__main__":
    unittest.main()
