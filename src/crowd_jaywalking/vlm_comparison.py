"""Compare supported VLMs on the same manually labelled context events."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from .config import ProjectConfig
from .jaad_context_evaluation import CONTEXT_FIELDS, JAADContextBenchmark


COMPARISON_FIELDS = (
    "selected",
    "model_id",
    "evaluated_events",
    "mean_context_macro_f1_percent",
    "mean_context_accuracy_percent",
    "policy_macro_f1_percent",
    "policy_accuracy_percent",
    "policy_coverage_percent",
    "result_directory",
)


def model_slug(model_id: str) -> str:
    """Return a readable collision resistant directory name for a model."""

    readable = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def comparison_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
    """Reduce one benchmark summary to model selection metrics."""

    context = summary["context_field_metrics"]
    macro_f1_values = [float(context[field]["macro_f1_percent"]) for field in CONTEXT_FIELDS]
    accuracy_values = [float(context[field]["accuracy_percent"]) for field in CONTEXT_FIELDS]
    policy = summary["policy_label_metrics"]
    return {
        "evaluated_events": int(summary["evaluated_events"]),
        "mean_context_macro_f1_percent": mean(macro_f1_values),
        "mean_context_accuracy_percent": mean(accuracy_values),
        "policy_macro_f1_percent": float(policy["macro_f1_percent"]),
        "policy_accuracy_percent": float(policy["accuracy_percent"]),
        "policy_coverage_percent": float(policy.get("coverage_percent", 0.0)),
    }


def select_model(rows: list[dict[str, Any]]) -> str:
    """Select the strongest observable context model with deterministic tie breaks."""

    if len(rows) < 2:
        raise ValueError("At least two completed model evaluations are required")
    sample_counts = {int(row["evaluated_events"]) for row in rows}
    if len(sample_counts) != 1:
        raise ValueError("Every comparison model must be evaluated on the same events")
    selected = max(
        rows,
        key=lambda row: (
            float(row["mean_context_macro_f1_percent"]),
            float(row["policy_macro_f1_percent"]),
            float(row["policy_coverage_percent"]),
            float(row["mean_context_accuracy_percent"]),
            str(row["model_id"]),
        ),
    )
    return str(selected["model_id"])


class VLMModelComparison:
    """Run each configured VLM sequentially and select one on development data."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.split = str(config.get("jaad_context_split")).strip().lower()
        settings = config.vlm_comparison_settings()
        self.model_ids = list(settings["models"])
        self.output_dir = Path(settings["results"]) / self.split
        self.comparison_csv = self.output_dir / "model_comparison.csv"
        self.summary_json = self.output_dir / "comparison_summary.json"
        self.selected_json = self.output_dir / "selected_model.json"

    def run(self) -> dict[str, Any]:
        """Evaluate all candidates and save a transparent selection record."""

        if self.split == "test":
            raise ValueError(
                "VLM model selection cannot use the locked test split. "
                "Set jaad_context_split to train or val."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        summaries: dict[str, dict[str, Any]] = {}
        for index, model_id in enumerate(self.model_ids, start=1):
            result_dir = self.output_dir / model_slug(model_id)
            print(f"\nVLM model {index}/{len(self.model_ids)}: {model_id}")
            summary = JAADContextBenchmark(
                self.config,
                model_id=model_id,
                output_dir=result_dir,
            ).run()
            summaries[model_id] = summary
            rows.append(
                {
                    "model_id": model_id,
                    **comparison_metrics(summary),
                    "result_directory": str(result_dir),
                }
            )

        selected_model = select_model(rows)
        for row in rows:
            row["selected"] = row["model_id"] == selected_model

        with self.comparison_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "selection_split": self.split,
            "selection_rule": (
                "Highest mean context macro F1, then policy macro F1, policy coverage, "
                "mean context accuracy, and model ID"
            ),
            "selected_model": selected_model,
            "models": rows,
            "model_summaries": summaries,
        }
        self.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.selected_json.write_text(
            json.dumps(
                {
                    "selected_model": selected_model,
                    "selection_split": self.split,
                    "config_entry": {"vlm_model": selected_model},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._print_summary(rows, selected_model)
        return payload

    @staticmethod
    def _print_summary(rows: list[dict[str, Any]], selected_model: str) -> None:
        print("\nJAAD VLM model comparison")
        for row in rows:
            marker = "SELECTED" if row["model_id"] == selected_model else ""
            print(
                f"{row['model_id']}: context macro F1="
                f"{float(row['mean_context_macro_f1_percent']):.2f}% "
                f"policy macro F1={float(row['policy_macro_f1_percent']):.2f}% "
                f"coverage={float(row['policy_coverage_percent']):.2f}% {marker}"
            )
        print(f"Selected model: {selected_model}")

