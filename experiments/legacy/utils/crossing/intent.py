import numpy as np
from typing import List, Optional, Dict, Any

LOW_MOTION_THRESH = 0.02


def detect_hesitation(
    stride_series: List[float],
    on_road_mask: List[bool],
    fps: float,
    low_motion_thresh: float = LOW_MOTION_THRESH,
    window_s: float = 0.5,
) -> bool:
    if not stride_series or not on_road_mask or fps <= 0:
        return False
    window = max(1, int(round(window_s * fps)))
    on_road_indices = [i for i, m in enumerate(on_road_mask) if m]
    if len(on_road_indices) < window:
        return False
    for i in range(len(on_road_indices) - window + 1):
        idxs = on_road_indices[i : i + window]
        ratios = [stride_series[j] for j in idxs if j < len(stride_series)]
        if len(ratios) < window:
            continue
        if all(r < low_motion_thresh for r in ratios):
            return True
    return False


def detect_inattentive_entry(facing_at_entry: str) -> bool:
    return facing_at_entry in ("SIDE_VIEW", "BACK_VIEW")


def classify_crossing(
    track_df=None,
    light_states: Optional[Dict[int, str]] = None,
    zebra_result: bool = False,
    fps: float = 30.0,
    hesitated: bool = False,
    inattentive_entry: bool = False,
    temporal_states: Optional[np.ndarray] = None,
    frame_count_start: int = 0,
) -> Dict[str, Any]:
    result = {
        "violation": False,
        "violation_type": None,
        "violation_confidence": 0.0,
        "risk_factors": [],
        "committed_frame": None,
    }
    COMMITTED = 2
    committed_frame = None
    if temporal_states is not None:
        committed_idxs = np.where(temporal_states == COMMITTED)[0]
        if len(committed_idxs) > 0:
            committed_frame = int(committed_idxs[0]) + frame_count_start

    result["committed_frame"] = committed_frame

    signal_violation = False
    no_crosswalk = False

    if committed_frame is not None and light_states:
        relevant = {f: s for f, s in light_states.items() if f >= committed_frame}
        if any(s == "RED" for s in relevant.values()):
            signal_violation = True

    if committed_frame is not None and zebra_result is False:
        no_crosswalk = True

    violations = []
    violation_score = 0.0
    if signal_violation:
        violations.append("SIGNAL_VIOLATION")
        violation_score += 0.9
    if no_crosswalk:
        violations.append("NO_CROSSWALK")
        violation_score += 0.7

    risk_factors = []
    if hesitated:
        risk_factors.append("HESITATION")
    if inattentive_entry:
        risk_factors.append("INATTENTIVE_ENTRY")

    violation_score = min(violation_score, 1.0)
    result["violation"] = violation_score >= 0.7
    result["violation_type"] = violations[0] if violations else None
    result["violation_confidence"] = violation_score
    result["risk_factors"] = risk_factors

    return result
