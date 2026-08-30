"""Local Hugging Face VLM inference for observable crossing context."""

from __future__ import annotations

import json
from typing import Any

from .models import ContextAssessment, EvidenceImage, Ternary, Visibility


PROMPT_VERSION = "global-context-v1"

CONTEXT_PROMPT = """You are inspecting one tracked pedestrian crossing event.

The images are chronological. For each time point, the full scene image is followed by an enlarged focus image.
The same target pedestrian is enclosed by a RED bounding box labelled TARGET PERSON in every image.

Report only observable scene facts that apply to this target person's crossing location. Do not decide whether the person is jaywalking. Do not use assumptions about a country or local law.

Return one JSON object with exactly these keys:
{
  "marked_crosswalk": "YES|NO|UNCERTAIN",
  "permissive_pedestrian_signal": "YES|NO|UNCERTAIN",
  "authorised_crossing_sign": "YES|NO|UNCERTAIN",
  "crossing_guard_permission": "YES|NO|UNCERTAIN",
  "prohibitive_pedestrian_signal": "YES|NO|UNCERTAIN",
  "visibility": "CLEAR|PARTIAL|INSUFFICIENT",
  "evidence_summary": "one short sentence describing visible evidence"
}

Definitions:
marked_crosswalk means visible zebra stripes or another clearly marked pedestrian crossing at the target's crossing path.
permissive_pedestrian_signal means a visible active walk or green pedestrian signal applying to the target.
authorised_crossing_sign means a visible sign explicitly designating the target location as a pedestrian crossing.
crossing_guard_permission means a visible authorised person is directing the target to cross.
prohibitive_pedestrian_signal means a visible red or do-not-walk pedestrian signal applying to the target.

Use UNCERTAIN when the relevant area is occluded, too small, outside the frame, or visually ambiguous. Output JSON only.
"""


class VLMError(RuntimeError):
    """Raised when local VLM inference cannot produce a valid assessment."""


class HuggingFaceContextClassifier:
    """Run Qwen2.5 VL locally with weights obtained from Hugging Face."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.model_id = str(settings["model_id"])
        self.max_new_tokens = int(settings.get("max_new_tokens", 300))
        self.model = None
        self.processor = None
        self._process_vision_info = None
        self._load(settings)

    def _load(self, settings: dict[str, Any]) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise VLMError(
                "Hugging Face VLM dependencies are missing. Run 'uv sync' from the repository root."
            ) from error

        cache_dir_value = settings.get("cache_dir")
        cache_dir = str(cache_dir_value) if cache_dir_value else None
        local_files_only = bool(settings.get("local_files_only", False))
        dtype = self._resolve_dtype(str(settings.get("torch_dtype", "auto")), torch)

        processor_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "min_pixels": int(settings.get("min_pixels", 200704)),
            "max_pixels": int(settings.get("max_pixels", 401408)),
        }
        model_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "device_map": settings.get("device_map", "auto"),
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }
        attention = settings.get("attn_implementation")
        if attention:
            model_kwargs["attn_implementation"] = str(attention)

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            self.model.eval()
        except Exception as error:
            raise VLMError(
                f"Could not load '{self.model_id}' from Hugging Face. "
                "Check disk space, memory, network access, and any required Hugging Face login."
            ) from error

        self._process_vision_info = process_vision_info

    def ensure_ready(self) -> None:
        """Confirm that the processor and model finished loading."""

        if self.processor is None or self.model is None or self._process_vision_info is None:
            raise VLMError(f"Hugging Face model '{self.model_id}' is not ready")

    def classify(self, evidence: list[EvidenceImage]) -> ContextAssessment:
        """Return a validated structured context assessment."""

        if not evidence:
            raise VLMError("No evidence images were supplied to the VLM")
        self.ensure_ready()

        content: list[dict[str, str]] = []
        for index, item in enumerate(evidence, start=1):
            content.extend(
                [
                    {"type": "text", "text": f"Time {index}: full scene."},
                    {"type": "image", "image": item.context_path.resolve().as_uri()},
                    {"type": "text", "text": f"Time {index}: target focus."},
                    {"type": "image", "image": item.focus_path.resolve().as_uri()},
                ]
            )
        content.append({"type": "text", "text": CONTEXT_PROMPT})
        messages = [{"role": "user", "content": content}]

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            import torch

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            trimmed_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as error:
            raise VLMError(
                f"Local Hugging Face inference failed for '{self.model_id}'. "
                "The evaluation stopped instead of inventing a label."
            ) from error

        return self._validate_response(output)

    @staticmethod
    def _resolve_dtype(value: str, torch_module: Any) -> Any:
        normalised = value.strip().lower()
        if normalised == "auto":
            return "auto"
        allowed = {
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "float32": torch_module.float32,
        }
        if normalised not in allowed:
            raise VLMError(
                "vlm.torch_dtype must be one of: auto, float16, bfloat16, float32"
            )
        return allowed[normalised]

    @staticmethod
    def _validate_response(content: str) -> ContextAssessment:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise VLMError(f"VLM returned invalid JSON: {content[:300]}") from error

        expected_keys = {
            "marked_crosswalk",
            "permissive_pedestrian_signal",
            "authorised_crossing_sign",
            "crossing_guard_permission",
            "prohibitive_pedestrian_signal",
            "visibility",
            "evidence_summary",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise VLMError(f"VLM returned unexpected context keys: {payload}")

        try:
            return ContextAssessment(
                marked_crosswalk=Ternary(str(payload["marked_crosswalk"]).upper()),
                permissive_pedestrian_signal=Ternary(
                    str(payload["permissive_pedestrian_signal"]).upper()
                ),
                authorised_crossing_sign=Ternary(str(payload["authorised_crossing_sign"]).upper()),
                crossing_guard_permission=Ternary(str(payload["crossing_guard_permission"]).upper()),
                prohibitive_pedestrian_signal=Ternary(
                    str(payload["prohibitive_pedestrian_signal"]).upper()
                ),
                visibility=Visibility(str(payload["visibility"]).upper()),
                evidence_summary=str(payload["evidence_summary"]).strip(),
            )
        except (KeyError, ValueError, TypeError) as error:
            raise VLMError(f"VLM returned an invalid context schema: {payload}") from error
