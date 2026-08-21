import base64
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import requests


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encodes a BGR OpenCV image array into a base64 JPEG string."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot encode empty frame.")
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


class OllamaClient:
    """Client for local Ollama API inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/api/chat",
        model: str = "qwen2.5vl:7b",
        temperature: float = 0.0,
        seed: Optional[int] = 42,
        max_tokens: int = 300,
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def generate_chat(
        self,
        prompt: str,
        base64_images: Optional[Union[str, List[str]]] = None,
        num_ctx: int = 16384,
    ) -> str:
        """Sends text prompt and optional images to Ollama chat endpoint."""
        if base64_images is None:
            images_list = []
        elif isinstance(base64_images, str):
            images_list = [base64_images]
        else:
            images_list = list(base64_images)

        options: Dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": num_ctx,
        }
        if self.seed is not None:
            options["seed"] = self.seed

        msg_obj: Dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }
        if images_list:
            msg_obj["images"] = images_list

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [msg_obj],
            "stream": False,
            "options": options,
        }


        try:
            response = requests.post(self.base_url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to communicate with Ollama at {self.base_url}. Ensure Ollama daemon is running. Error: {e}"
            ) from e

