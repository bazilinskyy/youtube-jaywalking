"""Tests for annotations and conservative evaluation metrics."""

import tempfile
import unittest
from pathlib import Path

from crowd_jaywalking.evaluation import calculate_metrics, load_annotations
from crowd_jaywalking.models import DecisionLabel


class EvaluationTests(unittest.TestCase):
    def test_load_annotations_maps_yes_and_no(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            path.write_text(
                "video_id,filename,label,split\n"
                "a,a.mp4,Yes,development\n"
                "b,b.mp4,No,development\n"
                "c,c.mp4,Not Sure,excluded\n"
                "d,d.mp4,Yes,locked_test\n",
                encoding="utf-8",
            )
            annotations, excluded = load_annotations(path, "development")
        self.assertEqual(
            [item.ground_truth for item in annotations],
            [DecisionLabel.JAYWALKING, DecisionLabel.COMPLIANT],
        )
        self.assertEqual(excluded, 1)

    def test_uncertain_reduces_coverage_and_overall_accuracy(self) -> None:
        rows = [
            {"ground_truth": "JAYWALKING", "prediction": "JAYWALKING"},
            {"ground_truth": "COMPLIANT", "prediction": "COMPLIANT"},
            {"ground_truth": "JAYWALKING", "prediction": "UNCERTAIN"},
        ]
        metrics = calculate_metrics(rows)
        self.assertAlmostEqual(metrics["coverage_percent"], 200.0 / 3.0)
        self.assertAlmostEqual(metrics["overall_accuracy_percent"], 200.0 / 3.0)
        self.assertEqual(metrics["decided_accuracy_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
