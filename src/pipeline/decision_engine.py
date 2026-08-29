"""
Production decision logic implementing the frozen Exp57/Exp58 Refined Context Synergy Architecture.
"""

from typing import List, Tuple


class DecisionEngine:
    """
    Synthesizes VLM unanimous votes, kinematic tracking, multi-temporal road overlaps,
    and wide context verification into a final JAYWALKING or COMPLIANT verdict.
    """

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
        """
        Executes the exact frozen Exp57 production decision rules.
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
