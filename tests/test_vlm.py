"""Tests for strict structured VLM output validation."""

import json
import unittest

from crowd_jaywalking.models import Ternary, Visibility
from crowd_jaywalking.vlm import HuggingFaceContextClassifier, VLMError


def valid_payload() -> dict[str, str]:
    return {
        "marked_crosswalk": "NO",
        "permissive_pedestrian_signal": "UNCERTAIN",
        "authorised_crossing_sign": "NO",
        "crossing_guard_permission": "NO",
        "prohibitive_pedestrian_signal": "NO",
        "visibility": "PARTIAL",
        "evidence_summary": "The target is visible but the signal is distant.",
    }


class VLMValidationTests(unittest.TestCase):
    def test_accepts_exact_json_schema(self) -> None:
        result = HuggingFaceContextClassifier._validate_response(json.dumps(valid_payload()))
        self.assertEqual(result.marked_crosswalk, Ternary.NO)
        self.assertEqual(result.permissive_pedestrian_signal, Ternary.UNCERTAIN)
        self.assertEqual(result.visibility, Visibility.PARTIAL)

    def test_accepts_markdown_json_fence(self) -> None:
        content = f"```json\n{json.dumps(valid_payload())}\n```"
        result = HuggingFaceContextClassifier._validate_response(content)
        self.assertEqual(result.authorised_crossing_sign, Ternary.NO)

    def test_rejects_missing_or_extra_keys(self) -> None:
        payload = valid_payload()
        payload["decision"] = "JAYWALKING"
        with self.assertRaises(VLMError):
            HuggingFaceContextClassifier._validate_response(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
