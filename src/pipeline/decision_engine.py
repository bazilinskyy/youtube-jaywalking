"""Production decision logic implementing the frozen Exp57/Exp58 Refined Context Synergy Architecture.

This module provides the DecisionEngine class, which synthesizes visual classification
consensus, pedestrian kinematics, road segmentation overlap, and wide-scene context
verification into a deterministic JAYWALKING or COMPLIANT decision.
"""

from typing import List, Tuple


class DecisionEngine:
    """Synthesizes multimodal perceptual evidence into final crossing compliance verdicts."""

    def evaluate(
        self,
        votes: List[str],
        lateral_displacement: float,
        mean_y: float,
        track_duration_sec: float,
        static_road_overlap: float,
        crosswalk_status: str,
        road_structure_status: str,
        junction_status: str,
    ) -> Tuple[str, str]:
        """Executes the frozen Exp57/Exp58 production decision rules.

        The evaluation follows a hierarchical decision protocol:
            - If 3/3 VLM votes are JAYWALKING:
                1. Rule 1 (Driveway Apron Filter): If pedestrian base is at the extreme bottom (mean_y > 0.84),
                   track duration is long (>6.0s), and road overlap is low (<0.30), classify as COMPLIANT
                   (filters out stationary/slow pedestrians standing near vehicle bumpers).
                2. Rule 2 (Crosswalk Verification): If crosswalk verification confirms LEGAL_CROSSWALK,
                   classify as COMPLIANT.
                3. Rule 3 (Junction Legal Crossing): If junction verification confirms LEGAL_JUNCTION_CROSSING,
                   road structure is PUBLIC_STREET, and lateral displacement >= 0.70, classify as COMPLIANT.
                4. Rule 4 (Enclosed Private Space): If road structure is PRIVATE_ENCLOSED and mean_y > 0.82,
                   classify as COMPLIANT (filters out private driveways and indoor parking structures).
                5. Rule 5 (Default Jaywalking): Confirmed public roadway crossing -> JAYWALKING.
            - Otherwise (<3 unanimous votes):
                - Fast-Crossing Sprint Dash Fallback: If 2/3 votes are JAYWALKING, track duration <= 1.5s,
                  lateral displacement >= 0.15, and crosswalk_status == NO_CROSSWALK, classify as JAYWALKING.
                - Default: COMPLIANT.

        Args:
            votes: List of independent binary predictions across the 3 sampled keyframes.
            lateral_displacement: Maximum normalized horizontal displacement (0.0 to 1.0).
            mean_y: Average normalized vertical position of the pedestrian bounding box base (0.0 to 1.0).
            track_duration_sec: Duration in seconds the pedestrian was tracked.
            static_road_overlap: Drivable road segmentation overlap ratio (0.0 to 1.0).
            crosswalk_status: Context router output for marked crosswalk ('LEGAL_CROSSWALK' or 'NO_CROSSWALK').
            road_structure_status: Context router output for roadway ('PUBLIC_STREET' or 'PRIVATE_ENCLOSED').
            junction_status: Context router output ('LEGAL_JUNCTION_CROSSING' or 'UNREGULATED_MIDBLOCK').

        Returns:
            A tuple of (prediction, decision_reason):
                - prediction (str): 'JAYWALKING' or 'COMPLIANT'.
                - decision_reason (str): Explanation of the active decision path.
        """
        p_unanimous = "JAYWALKING" if votes.count("JAYWALKING") == 3 else "COMPLIANT"

        if p_unanimous == "JAYWALKING":
            # Rule 1: Driveway apron bumper edge filter
            if mean_y > 0.84 and track_duration_sec > 6.0 and static_road_overlap < 0.30:
                return "COMPLIANT", "Driveway apron bumper filter"

            # Rule 2: Explicit marked legal crosswalk
            if crosswalk_status == "LEGAL_CROSSWALK":
                return "COMPLIANT", "Marked crosswalk detected"

            # Rule 3: Legal intersection junction corner crossing
            if (
                junction_status == "LEGAL_JUNCTION_CROSSING"
                and road_structure_status == "PUBLIC_STREET"
                and lateral_displacement >= 0.70
            ):
                return "COMPLIANT", "Legal intersection junction crossing confirmed"

            # Rule 4: Enclosed private garage or parking apron
            if road_structure_status == "PRIVATE_ENCLOSED" and mean_y > 0.82:
                return "COMPLIANT", "Enclosed private/indoor space detected"

            # Rule 5: Confirmed public roadway crossing
            return "JAYWALKING", "Confirmed public roadway crossing (unanimous VLM + public street)"

        else:
            # High-speed sprint fallback (2/3 majority within <=1.5s envelope with positive displacement)
            if (
                votes.count("JAYWALKING") == 2
                and track_duration_sec <= 1.5
                and lateral_displacement >= 0.15
                and crosswalk_status == "NO_CROSSWALK"
            ):
                return "JAYWALKING", "Fast-crossing dash with 2/3 VLM majority"

            return "COMPLIANT", "Compliant consensus"
