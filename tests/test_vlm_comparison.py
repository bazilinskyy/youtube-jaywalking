"""Tests for deterministic VLM model comparison."""

import unittest

from crowd_jaywalking.vlm_comparison import (
    comparison_metrics,
    model_slug,
    select_model,
)


def benchmark_summary(context_f1: float, policy_f1: float, coverage: float) -> dict:
    fields = (
        "marked_crosswalk",
        "permissive_pedestrian_signal",
        "authorised_crossing_sign",
        "crossing_guard_permission",
        "prohibitive_pedestrian_signal",
    )
    return {
        "evaluated_events": 20,
        "context_field_metrics": {
            field: {
                "accuracy_percent": context_f1 + 5.0,
                "macro_f1_percent": context_f1,
            }
            for field in fields
        },
        "policy_label_metrics": {
            "accuracy_percent": policy_f1 + 5.0,
            "macro_f1_percent": policy_f1,
            "coverage_percent": coverage,
        },
    }


class VLMComparisonTests(unittest.TestCase):
    def test_comparison_metrics_averages_context_fields(self) -> None:
        result = comparison_metrics(benchmark_summary(70.0, 80.0, 90.0))
        self.assertEqual(result["evaluated_events"], 20)
        self.assertAlmostEqual(result["mean_context_macro_f1_percent"], 70.0)
        self.assertAlmostEqual(result["policy_coverage_percent"], 90.0)

    def test_selects_context_f1_before_policy_f1(self) -> None:
        weaker_context = {
            "model_id": "model-a",
            **comparison_metrics(benchmark_summary(70.0, 95.0, 100.0)),
        }
        stronger_context = {
            "model_id": "model-b",
            **comparison_metrics(benchmark_summary(75.0, 80.0, 80.0)),
        }
        self.assertEqual(select_model([weaker_context, stronger_context]), "model-b")

    def test_requires_equal_sample_counts(self) -> None:
        first = {
            "model_id": "model-a",
            **comparison_metrics(benchmark_summary(70.0, 70.0, 90.0)),
        }
        second = dict(first, model_id="model-b", evaluated_events=19)
        with self.assertRaises(ValueError):
            select_model([first, second])

    def test_model_slug_is_readable_and_collision_resistant(self) -> None:
        first = model_slug("owner/model-a")
        second = model_slug("owner_model-a")
        self.assertTrue(first.startswith("owner_model_a_"))
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
