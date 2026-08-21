import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.cv.tracker import CVJaywalkingDetector
from src.ensemble import arbitrate_decision
from src.vlm.detector import VLMJaywalkingDetector


class EnsembleJaywalkingDetector:
    """Combines VLM semantic reasoning with classical CV tracking."""

    def __init__(
        self,
        vlm_detector: Optional[VLMJaywalkingDetector] = None,
        cv_detector: Optional[CVJaywalkingDetector] = None,
    ) -> None:
        self.vlm = vlm_detector or VLMJaywalkingDetector()
        self.cv = cv_detector or CVJaywalkingDetector()

    def predict(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        t0 = time.time()
        vlm_res = self.vlm.predict(video_path)
        cv_res = self.cv.predict(video_path)

        final_pred, conf, reason = arbitrate_decision(vlm_res, cv_res)
        elapsed = round(time.time() - t0, 3)

        return {
            "prediction": final_pred,
            "confidence": conf,
            "reason": reason,
            "vlm_result": vlm_res,
            "cv_result": cv_res,
            "elapsed_seconds": elapsed,
        }


def get_pipeline(
    mode: str = "vlm",
    prompt_name: str = "canonical",
    use_boundary_context: bool = False,
    use_pedestrian_motion: bool = False,
    use_vehicle_context: bool = False,
    min_votes: Optional[int] = None,
) -> Any:
    """
    Factory to return the selected inference pipeline.
    Modes:
        - 'vlm' / 'balanced' (default canonical VLM pipeline, min_votes=2)
        - 'high_precision' (unanimous 3/3 VLM votes, 82.05% Acc, 87.50% Spec)
        - 'safety' / 'high_recall' (sensitive VLM votes, min_votes=1, 100% Recall)
        - 'cv' (classical CV baseline)
        - 'ensemble' (fused VLM + CV decision)
    """
    mode = mode.lower()
    
    # Resolve vote threshold
    if min_votes is not None:
        resolved_min_votes = min_votes
    elif mode == "high_precision":
        resolved_min_votes = 3
    elif mode in ("safety", "high_recall"):
        resolved_min_votes = 1
    else:
        resolved_min_votes = 2

    if mode in ("vlm", "balanced", "high_precision", "safety", "high_recall"):
        return VLMJaywalkingDetector(
            prompt_name=prompt_name,
            use_boundary_context=use_boundary_context,
            use_pedestrian_motion=use_pedestrian_motion,
            use_vehicle_context=use_vehicle_context,
            min_votes_for_jaywalking=resolved_min_votes,
        )
    elif mode == "cv":
        return CVJaywalkingDetector()
    elif mode == "ensemble":
        return EnsembleJaywalkingDetector(
            vlm_detector=VLMJaywalkingDetector(
                prompt_name=prompt_name,
                use_boundary_context=use_boundary_context,
                use_pedestrian_motion=use_pedestrian_motion,
                use_vehicle_context=use_vehicle_context,
                min_votes_for_jaywalking=resolved_min_votes,
            )
        )
    elif mode in ("alpamayo", "full_video"):
        from src.vlm.alpamayo_detector import AlpamayoFullVideoDetector
        return AlpamayoFullVideoDetector()
    elif mode == "event_alpamayo":
        from src.vlm.alpamayo_detector import EventLocalizedAlpamayoDetector
        return EventLocalizedAlpamayoDetector()
    elif mode in ("alpamayo_gemma", "gemma_evaluator"):
        from src.vlm.gemma_evaluator import AlpamayoGemmaEvaluator
        return AlpamayoGemmaEvaluator()
    else:
        raise ValueError(
            f"Unknown pipeline mode: '{mode}'. Choose 'vlm', 'balanced', 'high_precision', 'safety', 'cv', 'ensemble', 'alpamayo', 'event_alpamayo', or 'alpamayo_gemma'."
        )





