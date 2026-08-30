"""Evaluate local VLM context predictions against manual JAAD annotations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .config import ProjectConfig
from .jaad_context import MANUAL_CONTEXT_FIELDS
from .models import DecisionLabel, EvidenceImage
from .policy import JaywalkingPolicy
from .vlm import CONTEXT_PROMPT, PROMPT_VERSION, HuggingFaceContextClassifier


CONTEXT_FIELDS = (
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
    "prohibitive_pedestrian_signal",
)

PREDICTION_FIELDS = (
    "video_id",
    "jaad_pedestrian_id",
    *(f"ground_truth_{field}" for field in CONTEXT_FIELDS),
    *(f"predicted_{field}" for field in CONTEXT_FIELDS),
    "ground_truth_visibility",
    "predicted_visibility",
    "ground_truth_label",
    "predicted_label",
    "policy_reason",
    "evidence_summary",
)


def _normalise_ternary(value: str) -> str:
    normalised = str(value).strip().upper()
    aliases = {"Y": "YES", "N": "NO", "U": "UNCERTAIN", "NOT SURE": "UNCERTAIN"}
    normalised = aliases.get(normalised, normalised)
    if normalised not in {"YES", "NO", "UNCERTAIN"}:
        raise ValueError(f"Context labels must be YES, NO, or UNCERTAIN: {value}")
    return normalised


def _normalise_visibility(value: str) -> str:
    normalised = str(value).strip().upper()
    if normalised not in {"CLEAR", "PARTIAL", "INSUFFICIENT"}:
        raise ValueError(
            f"Visibility must be CLEAR, PARTIAL, or INSUFFICIENT: {value}"
        )
    return normalised


def _normalise_jaywalking(value: str) -> str:
    normalised = str(value).strip().upper()
    mapping = {
        "YES": DecisionLabel.JAYWALKING.value,
        "JAYWALKING": DecisionLabel.JAYWALKING.value,
        "NO": DecisionLabel.COMPLIANT.value,
        "COMPLIANT": DecisionLabel.COMPLIANT.value,
        "UNCERTAIN": DecisionLabel.UNCERTAIN.value,
        "NOT SURE": DecisionLabel.UNCERTAIN.value,
    }
    if normalised not in mapping:
        raise ValueError(f"is_jaywalking must be YES, NO, or UNCERTAIN: {value}")
    return mapping[normalised]


def _macro_metrics(truth: list[str], predictions: list[str], labels: tuple[str, ...]) -> dict[str, float | int]:
    if len(truth) != len(predictions):
        raise ValueError("Ground truth and prediction lengths differ")
    per_label_f1: list[float] = []
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(truth, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(truth, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(truth, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label_f1.append(
            2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    correct = sum(actual == predicted for actual, predicted in zip(truth, predictions))
    return {
        "samples": len(truth),
        "accuracy_percent": 100.0 * correct / len(truth) if truth else 0.0,
        "macro_f1_percent": 100.0 * mean(per_label_f1) if per_label_f1 else 0.0,
    }


class JAADContextBenchmark:
    """Run and evaluate the VLM only on independently labelled true crossings."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.split = str(config.get("jaad_context_split")).strip().lower()
        self.output_dir = config.path("jaad_context_results") / self.split
        self.annotations_csv = self.output_dir / "context_annotations.csv"
        self.predictions_csv = self.output_dir / "vlm_predictions.csv"
        self.summary_json = self.output_dir / "vlm_summary.json"
        self.manifest_json = self.output_dir / "vlm_manifest.json"

    def run(self) -> dict[str, Any]:
        annotation_rows = self._annotation_rows()
        complete, incomplete = self._complete_rows(annotation_rows)
        if not complete:
            raise ValueError(
                f"No complete manual context annotations were found in {self.annotations_csv}"
            )
        for row in complete:
            for field in CONTEXT_FIELDS:
                _normalise_ternary(row[field])
            _normalise_visibility(row["visibility"])
            _normalise_jaywalking(row["is_jaywalking"])
        self._prepare_manifest(complete)
        completed = self._existing_predictions()

        classifier: HuggingFaceContextClassifier | None = None
        policy = JaywalkingPolicy(self.config.policy_settings())
        write_header = not self.predictions_csv.exists() or self.predictions_csv.stat().st_size == 0
        with self.predictions_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            if write_header:
                writer.writeheader()

            for index, row in enumerate(complete, start=1):
                key = (row["video_id"], row["jaad_pedestrian_id"])
                if key in completed:
                    print(f"[{index:03d}/{len(complete):03d}] {key}: already completed")
                    continue
                if classifier is None:
                    classifier = HuggingFaceContextClassifier(self.config.vlm_settings())
                    classifier.ensure_ready()
                print(f"[{index:03d}/{len(complete):03d}] {key}")
                context = classifier.classify(self._evidence(row))
                predicted_label, reason = policy.decide(context)
                writer.writerow(
                    {
                        "video_id": row["video_id"],
                        "jaad_pedestrian_id": row["jaad_pedestrian_id"],
                        **{
                            f"ground_truth_{field}": _normalise_ternary(row[field])
                            for field in CONTEXT_FIELDS
                        },
                        **{
                            f"predicted_{field}": getattr(context, field).value
                            for field in CONTEXT_FIELDS
                        },
                        "ground_truth_visibility": _normalise_visibility(row["visibility"]),
                        "predicted_visibility": context.visibility.value,
                        "ground_truth_label": _normalise_jaywalking(row["is_jaywalking"]),
                        "predicted_label": predicted_label.value,
                        "policy_reason": reason,
                        "evidence_summary": context.evidence_summary,
                    }
                )
                handle.flush()

        prediction_rows = self._read_csv(self.predictions_csv)
        summary = self._summarise(prediction_rows, len(incomplete))
        with self.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._print_summary(summary)
        return summary

    def _annotation_rows(self) -> list[dict[str, str]]:
        if not self.annotations_csv.is_file():
            raise FileNotFoundError(
                f"Context annotation sheet not found: {self.annotations_csv}. "
                "Run prepare_jaad_context_audit.py first."
            )
        rows = self._read_csv(self.annotations_csv)
        required = set(MANUAL_CONTEXT_FIELDS).difference({"annotator", "notes"})
        missing = required.difference(rows[0] if rows else {})
        if missing:
            raise ValueError(
                "Context annotation sheet is missing columns: " + ", ".join(sorted(missing))
            )
        return rows

    @staticmethod
    def _complete_rows(
        rows: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        required = [*CONTEXT_FIELDS, "visibility", "is_jaywalking"]
        complete = [row for row in rows if all(str(row.get(field, "")).strip() for field in required)]
        incomplete = [row for row in rows if row not in complete]
        return complete, incomplete

    def _prepare_manifest(self, annotated_rows: list[dict[str, str]]) -> None:
        annotation_payload = [
            {
                "video_id": row["video_id"],
                "jaad_pedestrian_id": row["jaad_pedestrian_id"],
                **{
                    field: str(row.get(field, "")).strip().upper()
                    for field in (*CONTEXT_FIELDS, "visibility", "is_jaywalking")
                },
            }
            for row in sorted(
                annotated_rows,
                key=lambda item: (item["video_id"], item["jaad_pedestrian_id"]),
            )
        ]
        annotation_sha256 = hashlib.sha256(
            json.dumps(annotation_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        fingerprint = self.config.fingerprint(
            f"{PROMPT_VERSION}\n{CONTEXT_PROMPT}\njaad-context-v1\n{annotation_sha256}"
        )
        if self.manifest_json.is_file():
            existing = json.loads(self.manifest_json.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "The VLM context results use another configuration or prompt. "
                    "Choose a new jaad_context_results path before rerunning."
                )
            return
        payload = {
            "pipeline_version": "1.1.0",
            "prompt_version": PROMPT_VERSION,
            "fingerprint": fingerprint,
            "split": self.split,
            "annotated_events": len(annotated_rows),
            "annotation_sha256": annotation_sha256,
            "vlm_model": self.config.get("vlm_model"),
        }
        self.manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _existing_predictions(self) -> set[tuple[str, str]]:
        if not self.predictions_csv.is_file():
            return set()
        return {
            (row["video_id"], row["jaad_pedestrian_id"])
            for row in self._read_csv(self.predictions_csv)
        }

    def _evidence(self, row: dict[str, str]) -> list[EvidenceImage]:
        directory = self.output_dir / row["evidence_directory"]
        if not directory.is_dir():
            raise FileNotFoundError(f"Evidence directory not found: {directory}")
        evidence: list[EvidenceImage] = []
        for context_path in sorted(directory.glob("frame_*_context.jpg")):
            frame_text = context_path.name.split("_")[1]
            focus_path = directory / context_path.name.replace("_context.jpg", "_focus.jpg")
            if not focus_path.is_file():
                raise FileNotFoundError(f"Focus evidence image not found: {focus_path}")
            evidence.append(
                EvidenceImage(
                    frame_index=int(frame_text),
                    context_path=context_path,
                    focus_path=focus_path,
                )
            )
        if not evidence:
            raise FileNotFoundError(f"No evidence images found: {directory}")
        return evidence

    def _summarise(
        self,
        rows: list[dict[str, str]],
        incomplete_rows: int,
    ) -> dict[str, Any]:
        field_metrics = {
            field: _macro_metrics(
                [row[f"ground_truth_{field}"] for row in rows],
                [row[f"predicted_{field}"] for row in rows],
                ("YES", "NO", "UNCERTAIN"),
            )
            for field in CONTEXT_FIELDS
        }
        field_metrics["visibility"] = _macro_metrics(
            [row["ground_truth_visibility"] for row in rows],
            [row["predicted_visibility"] for row in rows],
            ("CLEAR", "PARTIAL", "INSUFFICIENT"),
        )
        label_metrics = _macro_metrics(
            [row["ground_truth_label"] for row in rows],
            [row["predicted_label"] for row in rows],
            (
                DecisionLabel.JAYWALKING.value,
                DecisionLabel.COMPLIANT.value,
                DecisionLabel.UNCERTAIN.value,
            ),
        )
        return {
            "split": self.split,
            "evaluated_events": len(rows),
            "incomplete_annotation_rows": incomplete_rows,
            "context_field_metrics": field_metrics,
            "policy_label_metrics": label_metrics,
        }

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        print("\nJAAD VLM context benchmark")
        print(f"Split: {summary['split']}")
        print(f"Evaluated events: {summary['evaluated_events']}")
        for field, metrics in summary["context_field_metrics"].items():
            print(
                f"{field}: accuracy={metrics['accuracy_percent']:.2f}% "
                f"macro-F1={metrics['macro_f1_percent']:.2f}%"
            )
        label = summary["policy_label_metrics"]
        print(
            f"Policy label: accuracy={label['accuracy_percent']:.2f}% "
            f"macro-F1={label['macro_f1_percent']:.2f}%"
        )
