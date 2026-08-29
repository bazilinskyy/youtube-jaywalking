"""Context-aware verification routers for legal crosswalks, public road structures, and junction crossings.

This module provides the ContextRouter class, which evaluates wide-scene visual context
to resolve ambiguous crossings, detect zebra markings, differentiate public roadways from
private enclosed garages, and verify intersection corner crossings.
"""

from typing import Tuple
import numpy as np
from src.perception.vlm_classifier import (
    CROSSWALK_VERIFIER_PROMPT,
    LEGAL_JUNCTION_VERIFIER_PROMPT,
    PUBLIC_ROADWAY_VERIFIER_PROMPT,
    VLMClassifier,
)
from src.utils.video_utils import encode_frame_to_base64


class ContextRouter:
    """Evaluates wide-scene visual context to resolve ambiguous crossings, private lots, and crosswalks."""

    def __init__(self, vlm_classifier: VLMClassifier) -> None:
        """Initializes the context router with a VLM interface.

        Args:
            vlm_classifier: Initialized VLMClassifier instance used to query specialized prompts.
        """
        self.vlm = vlm_classifier

    def verify_scene_context(self, context_frame: np.ndarray) -> Tuple[str, str, str]:
        """Queries specialized visual prompts on the wide context frame.

        Executes 3 targeted visual inspections:
            1. Crosswalk & Zebra Markings: 'LEGAL_CROSSWALK' vs 'NO_CROSSWALK'.
            2. Public Roadway vs Enclosed Private Lot: 'PUBLIC_STREET' vs 'PRIVATE_ENCLOSED'.
            3. Intersection / Junction Legal Crossing: 'LEGAL_JUNCTION_CROSSING' vs 'UNREGULATED_MIDBLOCK'.

        Args:
            context_frame: OpenCV BGR image representing the wide scene (typically the midpoint frame).

        Returns:
            A tuple of (crosswalk_status, road_structure_status, junction_status):
                - crosswalk_status (str): 'LEGAL_CROSSWALK' or 'NO_CROSSWALK'.
                - road_structure_status (str): 'PUBLIC_STREET' or 'PRIVATE_ENCLOSED'.
                - junction_status (str): 'LEGAL_JUNCTION_CROSSING' or 'UNREGULATED_MIDBLOCK'.
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
        resp_junc = (
            "LEGAL_JUNCTION_CROSSING" if "LEGAL_JUNCTION_CROSSING" in resp_junc_raw.upper() else "UNREGULATED_MIDBLOCK"
        )

        return resp_cw, resp_road, resp_junc
