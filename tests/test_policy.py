"""Tests for the deterministic jaywalking definition."""

import unittest

from crowd_jaywalking.models import ContextAssessment, DecisionLabel, Ternary, Visibility
from crowd_jaywalking.policy import JaywalkingPolicy


def context(
    *,
    crosswalk: Ternary = Ternary.NO,
    signal: Ternary = Ternary.NO,
    sign: Ternary = Ternary.NO,
    guard: Ternary = Ternary.NO,
    prohibitive: Ternary = Ternary.NO,
    visibility: Visibility = Visibility.CLEAR,
) -> ContextAssessment:
    return ContextAssessment(
        marked_crosswalk=crosswalk,
        permissive_pedestrian_signal=signal,
        authorised_crossing_sign=sign,
        crossing_guard_permission=guard,
        prohibitive_pedestrian_signal=prohibitive,
        visibility=visibility,
        evidence_summary="test",
    )


class JaywalkingPolicyTests(unittest.TestCase):
    def test_no_permission_is_jaywalking(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(context())
        self.assertEqual(label, DecisionLabel.JAYWALKING)

    def test_marked_crosswalk_is_compliant(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(context(crosswalk=Ternary.YES))
        self.assertEqual(label, DecisionLabel.COMPLIANT)

    def test_permissive_signal_is_compliant(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(context(signal=Ternary.YES))
        self.assertEqual(label, DecisionLabel.COMPLIANT)

    def test_red_signal_overrides_crosswalk_by_default(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(
            context(crosswalk=Ternary.YES, prohibitive=Ternary.YES)
        )
        self.assertEqual(label, DecisionLabel.JAYWALKING)

    def test_partial_absence_is_uncertain(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(context(visibility=Visibility.PARTIAL))
        self.assertEqual(label, DecisionLabel.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
