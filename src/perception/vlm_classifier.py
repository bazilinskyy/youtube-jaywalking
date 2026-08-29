"""Vision-Language Model (VLM) client for zero-shot classification and scene verification.

This module provides the VLMClassifier class and prompt constants used to interact with
the locally hosted Ollama daemon (Qwen2.5-VL-7B). It handles prompt formatting, base64 image
payload construction, timeout management, and graceful error fallbacks.
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
    """Interface to the local Ollama vision-language model service (Qwen2.5-VL-7B)."""

    def __init__(
        self,
        model_name: str = "qwen2.5vl:7b",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initializes the VLM client configuration.

        Args:
            model_name: Name of the Ollama model to query. Defaults to 'qwen2.5vl:7b'.
            api_base: Base URL for the Ollama REST API. Defaults to 'http://localhost:11434'.
            temperature: Sampling temperature for deterministic generation. Defaults to 0.0.
            seed: Random seed for reproducible generation. Defaults to 42.
        """
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.seed = seed

    def query(self, prompt: str, base64_image: str, max_tokens: int = 30) -> str:
        """Sends a base64-encoded image and text prompt to the Ollama /api/chat endpoint.

        Args:
            prompt: Text prompt instructing the model on classification or scene verification.
            base64_image: JPEG image encoded as a base64 string.
            max_tokens: Maximum number of tokens to predict. Defaults to 30.

        Returns:
            The raw text response content from the model, stripped of leading/trailing whitespace.
            In case of network error or timeout, logs the error and returns 'COMPLIANT' as a safe default.
        """
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
