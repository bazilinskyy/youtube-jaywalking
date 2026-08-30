"""Deterministic country-independent jaywalking policy."""

from __future__ import annotations

from typing import Any

from .models import ContextAssessment, DecisionLabel, Ternary, Visibility


class JaywalkingPolicy:
    """Apply the operational definition to observable VLM context."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.prohibitive_overrides = bool(
            settings.get("prohibitive_signal_overrides_crosswalk", True)
        )

    def decide(self, context: ContextAssessment) -> tuple[DecisionLabel, str]:
        """Return the final label and a concise deterministic reason."""

        if context.visibility == Visibility.INSUFFICIENT:
            return DecisionLabel.UNCERTAIN, "Crossing infrastructure is not sufficiently visible"

        if (
            self.prohibitive_overrides
            and context.prohibitive_pedestrian_signal == Ternary.YES
        ):
            return DecisionLabel.JAYWALKING, "A prohibitive pedestrian signal is visibly active"

        permissions = {
            "marked crosswalk": context.marked_crosswalk,
            "permissive pedestrian signal": context.permissive_pedestrian_signal,
            "authorised crossing sign": context.authorised_crossing_sign,
            "crossing guard permission": context.crossing_guard_permission,
        }
        visible_permissions = [name for name, value in permissions.items() if value == Ternary.YES]
        if visible_permissions:
            return DecisionLabel.COMPLIANT, f"Visible permission: {', '.join(visible_permissions)}"

        if context.visibility == Visibility.PARTIAL:
            return DecisionLabel.UNCERTAIN, "No permission is visible, but scene visibility is partial"

        if any(value == Ternary.UNCERTAIN for value in permissions.values()):
            return DecisionLabel.UNCERTAIN, "At least one relevant crossing control is uncertain"

        return (
            DecisionLabel.JAYWALKING,
            "Valid road crossing without a marked crossing or permissive traffic control",
        )
