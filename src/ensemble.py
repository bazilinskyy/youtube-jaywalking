from typing import Any, Dict, Optional, Tuple


def arbitrate_decision(
    vlm_result: Dict[str, Any], cv_result: Optional[Dict[str, Any]] = None
) -> Tuple[str, str, str]:
    """
    Fuses predictions from VLM and classical CV branches.
    Returns:
        (final_prediction, confidence, reasoning)
    """
    vlm_pred = vlm_result.get("prediction", "unknown").lower()
    vlm_conf = vlm_result.get("confidence", "low")

    if cv_result is None:
        return vlm_pred, vlm_conf, f"VLM only: {vlm_result.get('reason', '')}"

    cv_pred = cv_result.get("prediction", "unknown").lower()
    cv_conf = cv_result.get("confidence", "low")

    # Consensus
    if vlm_pred == cv_pred and vlm_pred != "unknown":
        return vlm_pred, "high", "Consensus: VLM and CV pipeline agree"

    # High confidence VLM override
    if vlm_conf == "high":
        return vlm_pred, "high", f"VLM high confidence ({vlm_result.get('reason', '')})"

    # CV fallback if VLM is uncertain
    if cv_pred != "unknown" and cv_conf in ("high", "medium"):
        return cv_pred, "medium", f"CV tiebreaker: {cv_result.get('reason', '')}"

    return vlm_pred, vlm_conf, "VLM default"
