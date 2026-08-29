"""
Vision-Language Model (VLM) client for zero-shot and context-aware classification.
"""

import logging
import requests

logger = logging.getLogger(__name__)


CANONICAL_CLASSIFICATION_PROMPT = (
    "Analyze this video frame of a pedestrian crossing a road. "
    "Rules: GREEN light + crossing = COMPLIANT. "
    "Crossing sign + cars yielding = COMPLIANT. "
    "Zebra markings + pedestrian on them = COMPLIANT. "
    "No light + no sign + no crosswalk + on road = JAYWALKING. "
    "RED light + crossing = JAYWALKING. "
    "Classify as JAYWALKING or COMPLIANT. Reply with only one word."
)

CROSSWALK_VERIFIER_PROMPT = (
    "Carefully inspect this full driving scene for marked pedestrian crosswalks or traffic lights. "
    "Is the pedestrian crossing on white zebra stripes, a marked pedestrian crosswalk, or crossing legally at an "
    "intersection? Answer strictly with either 'LEGAL_CROSSWALK' or 'NO_CROSSWALK' followed by a one-sentence "
    "visual justification."
)

PUBLIC_ROADWAY_VERIFIER_PROMPT = (
    "Carefully inspect this road scene. Is this a public vehicle roadway (including residential streets, suburban "
    "roads, two-lane city streets) where through-traffic drives, OR is it strictly an enclosed indoor parking "
    "garage, private driveway apron, or pedestrian-only plaza? Answer strictly with either 'PUBLIC_STREET' or "
    "'PRIVATE_ENCLOSED' followed by a brief reason."
)

LEGAL_JUNCTION_VERIFIER_PROMPT = (
    "Examine this intersection or street crossing. Is the pedestrian crossing at an intersection corner, marked "
    "crosswalk, zebra crossing, with a pedestrian walk signal, or where traffic is yielding at a junction? "
    "Answer strictly with either 'LEGAL_JUNCTION_CROSSING' or 'UNREGULATED_MIDBLOCK' followed by a brief reason."
)


class VLMClassifier:
    """
    Interface to local Ollama VLM daemon (Qwen2.5-VL-7B).
    """

    def __init__(
        self,
        model_name: str = "qwen2.5vl:7b",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.seed = seed

    def query(self, prompt: str, base64_image: str, max_tokens: int = 30) -> str:
        """Sends an image and text prompt to the Ollama /api/chat endpoint."""
        url = f"{self.api_base}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image],
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"VLM inference request failed: {e}")
            return "COMPLIANT"
