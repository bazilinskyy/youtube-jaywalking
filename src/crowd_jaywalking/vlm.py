"""Local Hugging Face VLM inference for observable crossing context."""

from __future__ import annotations

import gc
import json
from typing import Any

from .models import ContextAssessment, EvidenceImage, Ternary, Visibility


PROMPT_VERSION = "global-context-v2"

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

Vehicle traffic lights are not pedestrian signals. Do not treat a red vehicle light as pedestrian permission or a green vehicle light as a prohibitive pedestrian signal.

Use NO only when the relevant crossing path or control is sufficiently visible and the feature is not present. Use UNCERTAIN when the relevant area is occluded, too small, outside the frame, or visually ambiguous. Output JSON only.
"""


class VLMError(RuntimeError):
    """Raised when local VLM inference cannot produce a valid assessment."""


class HuggingFaceContextClassifier:
    """Run a supported local VLM with weights obtained from Hugging Face."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.model_id = str(settings["model_id"])
        self.model_family = self._model_family(self.model_id)
        self.max_new_tokens = int(settings.get("max_new_tokens", 300))
        self.model = None
        self.processor = None
        self._process_vision_info = None
        self._load(settings)

    def _load(self, settings: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoProcessor
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
        }
        model_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "device_map": settings.get("device_map", "auto"),
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }
        attention = settings.get("attn_implementation")
        if attention:
            model_kwargs["attn_implementation"] = str(attention)

        try:
            if self.model_family == "qwen":
                from qwen_vl_utils import process_vision_info

                processor_kwargs.update(
                    {
                        "min_pixels": int(settings.get("min_pixels", 200704)),
                        "max_pixels": int(settings.get("max_pixels", 401408)),
                    }
                )
                if "qwen3" in self.model_id.lower():
                    from transformers import Qwen3VLForConditionalGeneration

                    model_class = Qwen3VLForConditionalGeneration
                else:
                    from transformers import Qwen2_5_VLForConditionalGeneration

                    model_class = Qwen2_5_VLForConditionalGeneration
                self._process_vision_info = process_vision_info
            else:
                from transformers import AutoModelForMultimodalLM

                model_class = AutoModelForMultimodalLM
                processor_kwargs["padding_side"] = "left"

            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
            self.model = model_class.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            self.model.eval()
        except Exception as error:
            raise VLMError(
                f"Could not load '{self.model_id}' from Hugging Face. "
                "Check disk space, memory, network access, and any required Hugging Face login."
            ) from error

    def ensure_ready(self) -> None:
        """Confirm that the processor and model finished loading."""

        if self.processor is None or self.model is None:
            raise VLMError(f"Hugging Face model '{self.model_id}' is not ready")
        if self.model_family == "qwen" and self._process_vision_info is None:
            raise VLMError(f"Qwen vision utilities for '{self.model_id}' are not ready")

    def classify(self, evidence: list[EvidenceImage]) -> ContextAssessment:
        """Return a validated structured context assessment."""

        if not evidence:
            raise VLMError("No evidence images were supplied to the VLM")
        self.ensure_ready()

        if self.model_family == "qwen":
            output = self._classify_qwen(evidence)
        else:
            output = self._classify_gemma(evidence)
        return self._validate_response(output)

    def _classify_qwen(self, evidence: list[EvidenceImage]) -> str:
        """Run Qwen2.5 VL or Qwen3 VL using its multimodal utility."""

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

        return output

    def _classify_gemma(self, evidence: list[EvidenceImage]) -> str:
        """Run Gemma 4 with images placed before the classification prompt."""

        content: list[dict[str, str]] = []
        image_order: list[str] = []
        for index, item in enumerate(evidence, start=1):
            content.extend(
                [
                    {"type": "image", "url": item.context_path.resolve().as_uri()},
                    {"type": "image", "url": item.focus_path.resolve().as_uri()},
                ]
            )
            image_order.append(
                f"Images {2 * index - 1} and {2 * index} are time {index}: "
                "full scene followed by target focus."
            )
        content.append(
            {
                "type": "text",
                "text": "\n".join([*image_order, CONTEXT_PROMPT]),
            }
        )
        messages = [{"role": "user", "content": content}]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[-1]

            import torch

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            return self.processor.decode(
                generated_ids[0][input_length:],
                skip_special_tokens=True,
            )
        except Exception as error:
            raise VLMError(
                f"Local Hugging Face inference failed for '{self.model_id}'. "
                "The evaluation stopped instead of inventing a label."
            ) from error

    def close(self) -> None:
        """Release model memory before another comparison model is loaded."""

        self.model = None
        self.processor = None
        self._process_vision_info = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _model_family(model_id: str) -> str:
        normalised = model_id.strip().lower()
        if "qwen2.5-vl" in normalised or "qwen3-vl" in normalised:
            return "qwen"
        if "gemma-4" in normalised:
            return "gemma"
        raise VLMError(
            "Unsupported VLM. Use a Qwen2.5 VL, Qwen3 VL, or Gemma 4 model ID."
        )

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
