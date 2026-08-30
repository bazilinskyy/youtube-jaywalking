"""Stage specific validation against official JAAD pedestrian annotations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from .config import ProjectConfig
from .crossing import CrossingDetectionResult, CrossingDetector
from .jaad import JAADDataset, JAADPedestrianTrack, JAADVideoAnnotations
from .models import BoundingBox, CrossingEvent, TrackObservation
from .tracking import PersonTracker, load_observations_csv, save_observations_csv


PERSON_CLASS = 0

PER_PEDESTRIAN_FIELDS = (
    "video_id",
    "pedestrian_id",
    "ground_truth_crossing",
    "track_matched",
    "matched_track_id",
    "visible_ground_truth_frames",
    "matched_iou_frames",
    "track_coverage",
    "mean_matched_iou",
    "predicted_crossing",
    "candidate_outcome",
    "ground_truth_start_frame",
    "ground_truth_end_frame",
    "predicted_transition_start_frame",
    "predicted_transition_end_frame",
    "transition_temporal_iou",
    "correct",
)

UNMATCHED_PREDICTION_FIELDS = (
    "video_id",
    "track_id",
    "start_frame",
    "end_frame",
    "transition_start_frame",
    "transition_end_frame",
    "outcome",
)


@dataclass(frozen=True)
class TrackMatch:
    """One independent JAAD pedestrian to YOLO track association."""

    pedestrian_id: str
    track_id: int
    visible_ground_truth_frames: int
    matched_iou_frames: int
    coverage: float
    mean_matched_iou: float


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate intersection over union for two normalised boxes."""

    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = first.width * first.height
    second_area = second.width * second.height
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _person_tracks(
    observations: list[TrackObservation],
) -> dict[int, dict[int, TrackObservation]]:
    tracks: dict[int, dict[int, TrackObservation]] = {}
    for observation in observations:
        if observation.class_id != PERSON_CLASS:
            continue
        track = tracks.setdefault(observation.track_id, {})
        previous = track.get(observation.frame_index)
        if previous is None or observation.confidence > previous.confidence:
            track[observation.frame_index] = observation
    return tracks


def match_person_tracks(
    annotations: JAADVideoAnnotations,
    observations: list[TrackObservation],
    iou_threshold: float,
    min_match_frames: int,
    min_track_coverage: float,
) -> tuple[dict[str, TrackMatch], set[int]]:
    """Greedily create one to one matches using frame aligned box IoU."""

    predicted = _person_tracks(observations)
    candidates: list[tuple[int, float, float, str, int, int]] = []

    for ground_truth in annotations.behaviour_tracks:
        visible_frames = ground_truth.visible_frames
        if not visible_frames:
            continue
        for track_id, predicted_by_frame in predicted.items():
            matched_ious: list[float] = []
            for frame in visible_frames:
                if frame not in predicted_by_frame:
                    continue
                overlap = box_iou(ground_truth.boxes[frame], predicted_by_frame[frame].box)
                if overlap >= float(iou_threshold):
                    matched_ious.append(overlap)
            matched_frames = len(matched_ious)
            coverage = matched_frames / len(visible_frames)
            if matched_frames < int(min_match_frames) or coverage < float(min_track_coverage):
                continue
            mean_iou = mean(matched_ious) if matched_ious else 0.0
            candidates.append(
                (
                    matched_frames,
                    coverage,
                    mean_iou,
                    ground_truth.pedestrian_id,
                    track_id,
                    len(visible_frames),
                )
            )

    candidates.sort(reverse=True)
    matches: dict[str, TrackMatch] = {}
    used_track_ids: set[int] = set()
    for matched_frames, coverage, mean_iou, pedestrian_id, track_id, visible_frames in candidates:
        if pedestrian_id in matches or track_id in used_track_ids:
            continue
        matches[pedestrian_id] = TrackMatch(
            pedestrian_id=pedestrian_id,
            track_id=track_id,
            visible_ground_truth_frames=visible_frames,
            matched_iou_frames=matched_frames,
            coverage=coverage,
            mean_matched_iou=mean_iou,
        )
        used_track_ids.add(track_id)

    return matches, set(predicted).difference(used_track_ids)


def _event_by_track(events: list[CrossingEvent]) -> dict[int, list[CrossingEvent]]:
    result: dict[int, list[CrossingEvent]] = {}
    for event in events:
        result.setdefault(event.person_id, []).append(event)
    return result


def _temporal_iou(ground_truth_frames: tuple[int, ...], event: CrossingEvent | None) -> float:
    if not ground_truth_frames or event is None:
        return 0.0
    ground_truth = set(ground_truth_frames)
    predicted = set(range(event.transition_start_frame, event.transition_end_frame + 1))
    union = ground_truth.union(predicted)
    return len(ground_truth.intersection(predicted)) / len(union) if union else 0.0


def _best_event(
    events: list[CrossingEvent],
    ground_truth_frames: tuple[int, ...],
) -> CrossingEvent | None:
    if not events:
        return None
    if not ground_truth_frames:
        return min(events, key=lambda item: (item.transition_start_frame, item.person_id))
    return max(events, key=lambda item: _temporal_iou(ground_truth_frames, item))


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(row["ground_truth_crossing"] and row["predicted_crossing"] for row in rows)
    tn = sum(not row["ground_truth_crossing"] and not row["predicted_crossing"] for row in rows)
    fp = sum(not row["ground_truth_crossing"] and row["predicted_crossing"] for row in rows)
    fn = sum(row["ground_truth_crossing"] and not row["predicted_crossing"] for row in rows)

    def percentage(numerator: float, denominator: float) -> float:
        return 100.0 * numerator / denominator if denominator else 0.0

    precision = percentage(tp, tp + fp)
    recall = percentage(tp, tp + fn)
    specificity = percentage(tn, tn + fp)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced = (recall + specificity) / 2.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_percent": percentage(tp + tn, len(rows)),
        "precision_percent": precision,
        "recall_percent": recall,
        "specificity_percent": specificity,
        "balanced_accuracy_percent": balanced,
        "f1_percent": f1,
    }


class JAADCrossingBenchmark:
    """Run tracking and crossing validation without loading the VLM."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.dataset = JAADDataset(config.path("jaad_root"))
        self.split = str(config.get("jaad_benchmark_split")).strip().lower()
        self.output_dir = config.path("jaad_benchmark_results") / self.split
        self.tracks_dir = self.output_dir / "tracks"
        self.per_pedestrian_csv = self.output_dir / "per_pedestrian.csv"
        self.unmatched_csv = self.output_dir / "unmatched_crossing_predictions.csv"
        self.summary_json = self.output_dir / "summary.json"
        self.detector = CrossingDetector(config.crossing_settings())
        self._tracker: PersonTracker | None = None

    @property
    def tracker(self) -> PersonTracker:
        if self._tracker is None:
            self._tracker = PersonTracker(self.config.tracking_settings(), self.config.root)
        return self._tracker

    def run(self) -> dict[str, Any]:
        video_ids = self.dataset.video_ids(self.split)
        selected_video = os.environ.get("CROWD_JAYWALKING_JAAD_VIDEO_ID", "").strip()
        if selected_video:
            if selected_video not in video_ids:
                raise ValueError(
                    f"{selected_video} is not in the official JAAD {self.split} split"
                )
            video_ids = [selected_video]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        unmatched_rows: list[dict[str, Any]] = []
        for index, video_id in enumerate(video_ids, start=1):
            print(f"[{index:03d}/{len(video_ids):03d}] {video_id}")
            annotations = self.dataset.load_video(video_id)
            video_path = self.dataset.clip_path(video_id)
            fps, observations = self._observations(video_id, video_path)
            detections = self.detector.detect(observations, fps)
            video_rows, video_unmatched = self._evaluate_video(
                annotations,
                observations,
                detections,
            )
            rows.extend(video_rows)
            unmatched_rows.extend(video_unmatched)

        self._write_csv(self.per_pedestrian_csv, PER_PEDESTRIAN_FIELDS, rows)
        self._write_csv(self.unmatched_csv, UNMATCHED_PREDICTION_FIELDS, unmatched_rows)
        summary = self._summarise(video_ids, rows, unmatched_rows)
        with self.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._print_summary(summary)
        return summary

    def _observations(self, video_id: str, video_path: Path) -> tuple[float, list[TrackObservation]]:
        import cv2

        tracking_csv = self.tracks_dir / f"{video_id}.csv"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open JAAD video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.release()

        if tracking_csv.is_file():
            return fps, load_observations_csv(tracking_csv)
        tracked_fps, observations = self.tracker.track(video_path)
        save_observations_csv(tracking_csv, observations)
        return tracked_fps, observations

    def _evaluate_video(
        self,
        annotations: JAADVideoAnnotations,
        observations: list[TrackObservation],
        detections: CrossingDetectionResult,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        matches, unmatched_track_ids = match_person_tracks(
            annotations=annotations,
            observations=observations,
            iou_threshold=float(self.config.get("jaad_match_iou")),
            min_match_frames=int(self.config.get("jaad_min_match_frames")),
            min_track_coverage=float(self.config.get("jaad_min_track_coverage")),
        )
        valid = _event_by_track(detections.valid_events)
        rejected = _event_by_track(detections.rejected_events)

        rows: list[dict[str, Any]] = []
        for ground_truth in annotations.behaviour_tracks:
            match = matches.get(ground_truth.pedestrian_id)
            valid_events = valid.get(match.track_id, []) if match else []
            rejected_events = rejected.get(match.track_id, []) if match else []
            event = _best_event(valid_events, ground_truth.crossing_frames)
            if event is None:
                event = _best_event(rejected_events, ground_truth.crossing_frames)
            predicted_crossing = bool(valid_events)
            intervals = ground_truth.crossing_intervals()
            ground_truth_start = intervals[0][0] if intervals else ""
            ground_truth_end = intervals[-1][1] if intervals else ""

            if valid_events:
                outcome = "ACCEPTED"
            elif rejected_events:
                outcome = rejected_events[0].rejection_reason.value
            elif match:
                outcome = "NO_CROSSING_CANDIDATE"
            else:
                outcome = "NO_MATCHED_TRACK"

            rows.append(
                {
                    "video_id": annotations.video_id,
                    "pedestrian_id": ground_truth.pedestrian_id,
                    "ground_truth_crossing": ground_truth.is_crossing,
                    "track_matched": match is not None,
                    "matched_track_id": match.track_id if match else "",
                    "visible_ground_truth_frames": len(ground_truth.visible_frames),
                    "matched_iou_frames": match.matched_iou_frames if match else 0,
                    "track_coverage": round(match.coverage, 6) if match else 0.0,
                    "mean_matched_iou": round(match.mean_matched_iou, 6) if match else 0.0,
                    "predicted_crossing": predicted_crossing,
                    "candidate_outcome": outcome,
                    "ground_truth_start_frame": ground_truth_start,
                    "ground_truth_end_frame": ground_truth_end,
                    "predicted_transition_start_frame": event.transition_start_frame if event else "",
                    "predicted_transition_end_frame": event.transition_end_frame if event else "",
                    "transition_temporal_iou": round(
                        _temporal_iou(ground_truth.crossing_frames, event),
                        6,
                    ),
                    "correct": ground_truth.is_crossing == predicted_crossing,
                }
            )

        unmatched_rows: list[dict[str, Any]] = []
        for event in detections.valid_events:
            if event.person_id not in unmatched_track_ids:
                continue
            unmatched_rows.append(
                {
                    "video_id": annotations.video_id,
                    "track_id": event.person_id,
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "transition_start_frame": event.transition_start_frame,
                    "transition_end_frame": event.transition_end_frame,
                    "outcome": "REQUIRES_MANUAL_REVIEW",
                }
            )
        return rows, unmatched_rows

    def _summarise(
        self,
        video_ids: list[str],
        rows: list[dict[str, Any]],
        unmatched_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        matched_rows = [row for row in rows if row["track_matched"]]
        true_positives = [
            row
            for row in rows
            if row["ground_truth_crossing"] and row["predicted_crossing"]
        ]
        return {
            "split": self.split,
            "videos": len(video_ids),
            "behaviour_annotated_pedestrians": len(rows),
            "ground_truth_crossing_pedestrians": sum(
                bool(row["ground_truth_crossing"]) for row in rows
            ),
            "matched_pedestrian_tracks": len(matched_rows),
            "track_match_recall_percent": 100.0 * len(matched_rows) / len(rows) if rows else 0.0,
            "mean_track_coverage_percent": (
                100.0 * mean(float(row["track_coverage"]) for row in matched_rows)
                if matched_rows
                else 0.0
            ),
            "mean_matched_iou": (
                mean(float(row["mean_matched_iou"]) for row in matched_rows)
                if matched_rows
                else 0.0
            ),
            "end_to_end_crossing_metrics": _classification_metrics(rows),
            "crossing_metrics_when_track_matched": _classification_metrics(matched_rows),
            "mean_transition_temporal_iou_on_true_positives": (
                mean(float(row["transition_temporal_iou"]) for row in true_positives)
                if true_positives
                else 0.0
            ),
            "unmatched_accepted_crossings_requiring_review": len(unmatched_rows),
            "configuration": {
                "match_iou": self.config.get("jaad_match_iou"),
                "min_match_frames": self.config.get("jaad_min_match_frames"),
                "min_track_coverage": self.config.get("jaad_min_track_coverage"),
            },
        }

    @staticmethod
    def _write_csv(
        path: Path,
        fields: tuple[str, ...],
        rows: list[dict[str, Any]],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        metrics = summary["end_to_end_crossing_metrics"]
        print("\nJAAD crossing benchmark")
        print(f"Split: {summary['split']}")
        print(f"Videos: {summary['videos']}")
        print(f"Annotated pedestrians: {summary['behaviour_annotated_pedestrians']}")
        print(f"Track match recall: {summary['track_match_recall_percent']:.2f}%")
        print(f"Crossing precision: {metrics['precision_percent']:.2f}%")
        print(f"Crossing recall: {metrics['recall_percent']:.2f}%")
        print(f"Crossing F1: {metrics['f1_percent']:.2f}%")
        print(f"Balanced accuracy: {metrics['balanced_accuracy_percent']:.2f}%")
