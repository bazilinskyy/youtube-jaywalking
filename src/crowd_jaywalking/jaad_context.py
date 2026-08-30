"""Prepare ground truth crossing evidence for manual VLM context annotation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .jaad import JAADDataset, JAADPedestrianTrack, JAADVideoAnnotations


MANUAL_CONTEXT_FIELDS = (
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
    "prohibitive_pedestrian_signal",
    "visibility",
    "is_jaywalking",
    "annotator",
    "notes",
)

CONTEXT_AUDIT_FIELDS = (
    "video_id",
    "filename",
    "jaad_pedestrian_id",
    "split",
    "crossing_start_frame",
    "crossing_end_frame",
    "jaad_designated",
    "jaad_signalized",
    "jaad_ped_crossing",
    "jaad_ped_sign",
    "jaad_traffic_light",
    *MANUAL_CONTEXT_FIELDS,
    "evidence_directory",
)


class JAADContextAuditBuilder:
    """Create person focused evidence and a non-destructive annotation template."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.dataset = JAADDataset(config.path("jaad_root"))
        self.split = str(config.get("jaad_context_split")).strip().lower()
        self.output_dir = config.path("jaad_context_results") / self.split
        self.evidence_dir = self.output_dir / "evidence"
        self.annotations_csv = self.output_dir / "context_annotations.csv"
        self.sample_positions = [float(value) for value in config.get("evidence_sample_positions")]
        self.context_seconds = float(config.get("evidence_context_seconds"))
        self.crop_margin = float(config.get("evidence_crop_margin"))
        self.max_dimension = int(config.get("evidence_max_dimension"))
        self.jpeg_quality = int(config.get("evidence_jpeg_quality"))

    def run(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        existing = self._existing_rows()
        rows: list[dict[str, Any]] = []
        video_ids = self.dataset.video_ids(self.split)

        for index, video_id in enumerate(video_ids, start=1):
            print(f"[{index:03d}/{len(video_ids):03d}] {video_id}")
            annotations = self.dataset.load_video(video_id)
            crossing_tracks = [
                track for track in annotations.behaviour_tracks if track.is_crossing
            ]
            if not crossing_tracks:
                continue
            video_path = self.dataset.clip_path(video_id)
            for track in crossing_tracks:
                evidence_directory = self._build_evidence(video_path, annotations, track)
                key = (video_id, track.pedestrian_id)
                row = self._row(annotations, track, evidence_directory)
                for field in MANUAL_CONTEXT_FIELDS:
                    row[field] = existing.get(key, {}).get(field, "")
                rows.append(row)

        with self.annotations_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CONTEXT_AUDIT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Crossing events prepared: {len(rows)}")
        print(f"Saved: {self.annotations_csv}")
        return self.annotations_csv

    def _existing_rows(self) -> dict[tuple[str, str], dict[str, str]]:
        if not self.annotations_csv.is_file():
            return {}
        with self.annotations_csv.open("r", encoding="utf-8", newline="") as handle:
            return {
                (str(row.get("video_id", "")), str(row.get("jaad_pedestrian_id", ""))): row
                for row in csv.DictReader(handle)
            }

    def _row(
        self,
        annotations: JAADVideoAnnotations,
        track: JAADPedestrianTrack,
        evidence_directory: Path,
    ) -> dict[str, Any]:
        crossing_frames = track.crossing_frames
        traffic = [
            annotations.traffic[frame]
            for frame in crossing_frames
            if frame in annotations.traffic
        ]

        def any_one(name: str) -> str:
            values = {item.get(name, "") for item in traffic}
            if "1" in values:
                return "1"
            if "0" in values:
                return "0"
            return ""

        traffic_lights = sorted(
            {
                item.get("traffic_light", "")
                for item in traffic
                if item.get("traffic_light", "")
            }
        )
        return {
            "video_id": annotations.video_id,
            "filename": f"{annotations.video_id}.mp4",
            "jaad_pedestrian_id": track.pedestrian_id,
            "split": self.split,
            "crossing_start_frame": crossing_frames[0],
            "crossing_end_frame": crossing_frames[-1],
            "jaad_designated": track.attributes.get("designated", ""),
            "jaad_signalized": track.attributes.get("signalized", ""),
            "jaad_ped_crossing": any_one("ped_crossing"),
            "jaad_ped_sign": any_one("ped_sign"),
            "jaad_traffic_light": "|".join(traffic_lights),
            **{field: "" for field in MANUAL_CONTEXT_FIELDS},
            "evidence_directory": str(evidence_directory.relative_to(self.output_dir)),
        }

    def _build_evidence(
        self,
        video_path: Path,
        annotations: JAADVideoAnnotations,
        track: JAADPedestrianTrack,
    ) -> Path:
        import cv2

        crossing_frames = track.crossing_frames
        event_directory = self.evidence_dir / annotations.video_id / track.pedestrian_id
        event_directory.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open JAAD video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        context_frames = max(0, int(round(self.context_seconds * fps)))
        evidence_start = max(track.frames[0], crossing_frames[0] - context_frames)
        evidence_end = min(track.frames[-1], crossing_frames[-1] + context_frames)
        candidate_frames = [
            frame for frame in track.frames if evidence_start <= frame <= evidence_end
        ]

        try:
            for position in self.sample_positions:
                requested = int(round(evidence_start + position * (evidence_end - evidence_start)))
                frame_index = min(candidate_frames, key=lambda frame: abs(frame - requested))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
                self._save_pair(image, track, frame_index, event_directory)
        finally:
            capture.release()
        return event_directory

    def _save_pair(
        self,
        image,
        track: JAADPedestrianTrack,
        frame_index: int,
        event_directory: Path,
    ) -> None:
        import cv2

        height, width = image.shape[:2]
        box = track.boxes[frame_index]
        x1 = max(0, min(width - 1, int(round(box.x1 * width))))
        y1 = max(0, min(height - 1, int(round(box.y1 * height))))
        x2 = max(x1 + 1, min(width, int(round(box.x2 * width))))
        y2 = max(y1 + 1, min(height, int(round(box.y2 * height))))

        context = image.copy()
        self._draw_target(context, x1, y1, x2, y2, track.pedestrian_id)
        context = self._resize(context)

        margin_x = int(round((x2 - x1) * self.crop_margin))
        margin_y = int(round((y2 - y1) * self.crop_margin))
        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(width, x2 + margin_x)
        crop_y2 = min(height, y2 + margin_y)
        focus = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        self._draw_target(
            focus,
            x1 - crop_x1,
            y1 - crop_y1,
            x2 - crop_x1,
            y2 - crop_y1,
            track.pedestrian_id,
        )
        focus = self._resize(focus)

        parameters = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        context_path = event_directory / f"frame_{frame_index:06d}_context.jpg"
        focus_path = event_directory / f"frame_{frame_index:06d}_focus.jpg"
        if not cv2.imwrite(str(context_path), context, parameters):
            raise RuntimeError(f"Could not save evidence image: {context_path}")
        if not cv2.imwrite(str(focus_path), focus, parameters):
            raise RuntimeError(f"Could not save evidence image: {focus_path}")

    @staticmethod
    def _draw_target(
        image,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        pedestrian_id: str,
    ) -> None:
        import cv2

        colour = (0, 0, 255)
        thickness = max(2, int(round(max(image.shape[:2]) / 400)))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(
            image,
            f"TARGET {pedestrian_id}",
            (max(0, x1), max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )

    def _resize(self, image):
        import cv2

        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.max_dimension:
            return image
        scale = self.max_dimension / longest
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
