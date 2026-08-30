"""Tests for manual JAAD context label validation and metrics."""

import unittest

from crowd_jaywalking.jaad_context_evaluation import (
    _macro_metrics,
    _normalise_jaywalking,
    _normalise_ternary,
    _normalise_visibility,
)


class JAADContextEvaluationTests(unittest.TestCase):
    def test_normalises_manual_labels(self) -> None:
        self.assertEqual(_normalise_ternary("yes"), "YES")
        self.assertEqual(_normalise_ternary("Not Sure"), "UNCERTAIN")
        self.assertEqual(_normalise_visibility("partial"), "PARTIAL")
        self.assertEqual(_normalise_jaywalking("no"), "COMPLIANT")

    def test_macro_metrics_reports_accuracy(self) -> None:
        result = _macro_metrics(
            ["YES", "NO", "UNCERTAIN"],
            ["YES", "YES", "UNCERTAIN"],
            ("YES", "NO", "UNCERTAIN"),
        )
        self.assertAlmostEqual(result["accuracy_percent"], 200.0 / 3.0)
        self.assertGreater(result["macro_f1_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
