"""CROWD style flat project configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTIVE_CONFIG_NAME = "config"
DEFAULT_CONFIG_NAME = "default.config"

OPTIONAL_CROSSING_DEFAULTS = {
    "partial_crossing_enabled": True,
    "partial_exit_min_x_range": 0.48,
}

CROSSING_KEYS = (
    "road_left",
    "road_right",
    "boundary_tolerance",
    "min_track_seconds",
    "min_road_seconds",
    "max_track_gap_seconds",
    "min_crossing_x_range",
    "max_crossing_speed_per_frame",
    "low_x_range",
    "low_x_min_road_seconds",
    "weak_x_range",
    "long_weak_road_seconds",
    "weak_y_jitter_x_range",
    "weak_y_jitter_motion",
    "weak_y_jitter_height",
    "jitter_road_seconds",
    "tiny_long_track_x_range",
    "tiny_long_track_height",
    "tiny_long_track_road_seconds",
    "tiny_no_static_height",
    "tiny_no_static_width",
    "tiny_no_static_min_road_seconds",
    "no_static_tiny_min_road_seconds",
    "no_static_tiny_fast_speed",
    "slender_track_width",
    "slender_track_height",
    "slender_track_min_road_seconds",
    "slender_track_max_road_seconds",
    "no_static_slender_height",
    "no_static_slender_max_road_seconds",
    "slender_static_min_relative_x_range",
    "large_lateral_x_range",
    "large_lateral_tiny_height",
    "min_static_shared_seconds",
    "camera_static_x_range",
    "camera_ratio_threshold",
    "camera_static_relative_x_range",
    "camera_static_height",
    "camera_static_tiny_relative_x_range",
    "camera_static_tiny_height",
    "camera_tiny_height",
    "camera_min_road_seconds",
    "min_relative_x_range",
    "rider_min_shared_seconds",
    "rider_min_continuous_shared_seconds",
    "rider_shared_run_gap_seconds",
    "rider_min_vehicle_width_ratio",
    "rider_min_vehicle_width_ratio_frames",
    "rider_distance_relative_threshold",
    "rider_proximity_ratio",
    "rider_alpha_x",
    "rider_beta_y",
    "rider_gamma_y",
    "rider_colocation_ratio",
    "rider_similarity_threshold",
    "rider_similarity_ratio",
    "rider_min_motion_seconds",
    "rider_motion_colocation_min",
    "rider_short_shared_seconds",
    "rider_short_similarity_ratio",
    "rider_short_displacement",
)

REQUIRED_KEYS = {
    "data",
    "videos",
    "source_annotations",
    "annotations",
    "evaluation_split",
    "results",
    "tracking_model",
    "bbox_tracker",
    "min_confidence",
    "iou",
    "device",
    *CROSSING_KEYS,
    "evidence_sample_positions",
    "evidence_context_seconds",
    "evidence_crop_margin",
    "evidence_max_dimension",
    "evidence_jpeg_quality",
    "vlm_model",
    "vlm_device_map",
    "vlm_torch_dtype",
    "vlm_attn_implementation",
    "vlm_cache_dir",
    "vlm_local_files_only",
    "vlm_min_pixels",
    "vlm_max_pixels",
    "vlm_max_new_tokens",
    "prohibitive_signal_overrides_crosswalk",
    "resume",
    "split_seed",
    "development_fraction",
    "validation_fraction",
    "locked_test_fraction",
    "jaad_root",
    "jaad_benchmark_split",
    "jaad_benchmark_results",
    "jaad_match_iou",
    "jaad_min_match_frames",
    "jaad_min_track_coverage",
    "jaad_context_split",
    "jaad_context_results",
}


@dataclass(frozen=True)
class ProjectConfig:
    """Validated flat project configuration and its source path."""

    source_path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProjectConfig":
        """Load an explicit config, or use config with default.config as fallback."""

        if path is None:
            active = (Path.cwd() / ACTIVE_CONFIG_NAME).resolve()
            fallback = (Path.cwd() / DEFAULT_CONFIG_NAME).resolve()
            source = active if active.is_file() else fallback
        else:
            source = Path(path).resolve()

        if not source.is_file():
            if path is None:
                raise FileNotFoundError(
                    f"Neither '{ACTIVE_CONFIG_NAME}' nor '{DEFAULT_CONFIG_NAME}' was found "
                    f"in {Path.cwd().resolve()}"
                )
            raise FileNotFoundError(f"Configuration file not found: {source}")

        try:
            with source.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Configuration is badly formatted: {source}. "
                f"Update it using '{DEFAULT_CONFIG_NAME}' as the template."
            ) from error

        if not isinstance(raw, dict):
            raise ValueError(f"Configuration root must be an object: {source}")

        config = cls(source_path=source, raw=raw)
        config.validate()
        return config

    @property
    def root(self) -> Path:
        return self.source_path.parent

    def get(self, name: str) -> Any:
        if name not in self.raw:
            raise KeyError(f"Missing configuration entry: {name}")
        return self.raw[name]

    def path(self, name: str) -> Path:
        value = self.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration entry must be a path string: {name}")
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    def paths(self, name: str) -> list[Path]:
        values = self.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Configuration entry must be a non-empty path list: {name}")
        resolved: list[Path] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Configuration path list contains an invalid value: {name}")
            candidate = Path(value)
            resolved.append(
                candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
            )
        return resolved

    def data_file(self, name: str) -> Path:
        value = self.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration entry must name a data file: {name}")
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.paths("data")[0] / candidate).resolve()

    def tracking_settings(self) -> dict[str, Any]:
        return {
            "model": self.get("tracking_model"),
            "tracker": self.get("bbox_tracker"),
            "confidence": self.get("min_confidence"),
            "iou": self.get("iou"),
            "device": self.get("device"),
        }

    def crossing_settings(self) -> dict[str, Any]:
        settings = {key: self.get(key) for key in CROSSING_KEYS}
        settings.update(
            {
                key: self.raw.get(key, default)
                for key, default in OPTIONAL_CROSSING_DEFAULTS.items()
            }
        )
        return settings

    def evidence_settings(self) -> dict[str, Any]:
        return {
            "sample_positions": self.get("evidence_sample_positions"),
            "context_seconds": self.get("evidence_context_seconds"),
            "crop_margin": self.get("evidence_crop_margin"),
            "max_dimension": self.get("evidence_max_dimension"),
            "jpeg_quality": self.get("evidence_jpeg_quality"),
        }

    def vlm_settings(self) -> dict[str, Any]:
        return {
            "model_id": self.get("vlm_model"),
            "device_map": self.get("vlm_device_map"),
            "torch_dtype": self.get("vlm_torch_dtype"),
            "attn_implementation": self.get("vlm_attn_implementation"),
            "cache_dir": self.get("vlm_cache_dir"),
            "local_files_only": self.get("vlm_local_files_only"),
            "min_pixels": self.get("vlm_min_pixels"),
            "max_pixels": self.get("vlm_max_pixels"),
            "max_new_tokens": self.get("vlm_max_new_tokens"),
        }

    def policy_settings(self) -> dict[str, Any]:
        return {
            "prohibitive_signal_overrides_crosswalk": self.get(
                "prohibitive_signal_overrides_crosswalk"
            )
        }

    def fingerprint(self, prompt_version: str) -> str:
        effective_config = dict(self.raw)
        for key, default in OPTIONAL_CROSSING_DEFAULTS.items():
            effective_config.setdefault(key, default)
        payload = {
            "config": effective_config,
            "prompt_version": prompt_version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        missing = sorted(REQUIRED_KEYS.difference(self.raw))
        if missing:
            raise ValueError(f"Missing configuration entries: {', '.join(missing)}")
        nested = sorted(key for key, value in self.raw.items() if isinstance(value, dict))
        if nested:
            raise ValueError(
                "Configuration must be flat; nested objects found at: " + ", ".join(nested)
            )

        self.paths("data")
        self.paths("videos")
        self.data_file("source_annotations")
        self.data_file("annotations")
        self.path("results")

        left = float(self.get("road_left"))
        right = float(self.get("road_right"))
        if not 0.0 <= left < right <= 1.0:
            raise ValueError("road_left and road_right must satisfy 0 <= left < right <= 1")
        partial_crossing_enabled = self.raw.get(
            "partial_crossing_enabled",
            OPTIONAL_CROSSING_DEFAULTS["partial_crossing_enabled"],
        )
        if not isinstance(partial_crossing_enabled, bool):
            raise ValueError("partial_crossing_enabled must be true or false")
        partial_exit_min_x_range = float(
            self.raw.get(
                "partial_exit_min_x_range",
                OPTIONAL_CROSSING_DEFAULTS["partial_exit_min_x_range"],
            )
        )
        if not 0.0 <= partial_exit_min_x_range <= 1.0:
            raise ValueError("partial_exit_min_x_range must be between 0 and 1")

        positions = self.get("evidence_sample_positions")
        if not isinstance(positions, list) or not positions:
            raise ValueError("evidence_sample_positions must be a non-empty list")
        if any(not 0.0 <= float(position) <= 1.0 for position in positions):
            raise ValueError("Every evidence sample position must be between 0 and 1")
        if float(self.get("evidence_context_seconds")) < 0.0:
            raise ValueError("evidence_context_seconds must be non-negative")

        if not str(self.get("tracking_model")).strip():
            raise ValueError("tracking_model must name a YOLO model")
        if not str(self.get("bbox_tracker")).strip():
            raise ValueError("bbox_tracker must name a tracker configuration")
        if not str(self.get("vlm_model")).strip():
            raise ValueError("vlm_model must name a Hugging Face model")

        min_pixels = int(self.get("vlm_min_pixels"))
        max_pixels = int(self.get("vlm_max_pixels"))
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("VLM pixel limits must satisfy 0 < min_pixels <= max_pixels")
        if int(self.get("vlm_max_new_tokens")) <= 0:
            raise ValueError("vlm_max_new_tokens must be positive")

        self.path("jaad_root")
        self.path("jaad_benchmark_results")
        self.path("jaad_context_results")
        for name in ("jaad_benchmark_split", "jaad_context_split"):
            if str(self.get(name)).strip().lower() not in {"train", "val", "test"}:
                raise ValueError(f"{name} must be one of: train, val, test")
        if not 0.0 < float(self.get("jaad_match_iou")) <= 1.0:
            raise ValueError("jaad_match_iou must be greater than 0 and at most 1")
        if int(self.get("jaad_min_match_frames")) <= 0:
            raise ValueError("jaad_min_match_frames must be positive")
        if not 0.0 < float(self.get("jaad_min_track_coverage")) <= 1.0:
            raise ValueError("jaad_min_track_coverage must be greater than 0 and at most 1")
