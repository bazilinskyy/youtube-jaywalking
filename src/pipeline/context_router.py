"""
Context-aware verification routers for legal crosswalks, public road structures, and junction crossings.
"""

from typing import Tuple
import numpy as np
from src.perception.vlm_classifier import (
    VLMClassifier,
    CROSSWALK_VERIFIER_PROMPT,
    PUBLIC_ROADWAY_VERIFIER_PROMPT,
    LEGAL_JUNCTION_VERIFIER_PROMPT,
)
from src.utils.video_utils import encode_frame_to_base64


class ContextRouter:
    """
    Evaluates wide-scene visual context to resolve ambiguous crossings, private lots, and crosswalks.
    """
    def __init__(self, vlm_classifier: VLMClassifier):
        self.vlm = vlm_classifier

    def verify_scene_context(self, context_frame: np.ndarray) -> Tuple[str, str, str]:
        """
        Queries specialized visual prompts on the wide context frame.
        Returns: (crosswalk_status, road_structure_status, junction_status)
        """
        b64 = encode_frame_to_base64(context_frame, quality=85)
        
        # 1. Crosswalk & Zebra Markings
        resp_cw_raw = self.vlm.query(CROSSWALK_VERIFIER_PROMPT, b64)
        resp_cw = "LEGAL_CROSSWALK" if "LEGAL_CROSSWALK" in resp_cw_raw.upper() else "NO_CROSSWALK"
        
        # 2. Public Roadway vs Enclosed Private Lot
        resp_road_raw = self.vlm.query(PUBLIC_ROADWAY_VERIFIER_PROMPT, b64)
        resp_road = "PUBLIC_STREET" if "PUBLIC_STREET" in resp_road_raw.upper() else "PRIVATE_ENCLOSED"
        
        # 3. Intersection / Junction Legal Crossing
        resp_junc_raw = self.vlm.query(LEGAL_JUNCTION_VERIFIER_PROMPT, b64)
        resp_junc = "LEGAL_JUNCTION_CROSSING" if "LEGAL_JUNCTION_CROSSING" in resp_junc_raw.upper() else "UNREGULATED_MIDBLOCK"
        
        return resp_cw, resp_road, resp_junc
