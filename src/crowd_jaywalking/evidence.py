"""Generate person-focused and full-scene visual evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import CrossingEvent, EvidenceImage, TrackObservation


class EvidenceBuilder:
    """Create chronological context and focus images for one crossing person."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.sample_positions = [float(value) for value in settings["sample_positions"]]
        self.context_seconds = float(settings.get("context_seconds", 0.50))
        self.crop_margin = float(settings.get("crop_margin", 0.75))
        self.max_dimension = int(settings.get("max_dimension", 1280))
        self.jpeg_quality = int(settings.get("jpeg_quality", 90))

    def build(
        self,
        video_path: str | Path,
        event: CrossingEvent,
        observations: list[TrackObservation],
        output_root: str | Path,
        fps: float,
    ) -> list[EvidenceImage]:
        """Save crossing-centred context and expanded target crops for one event."""

        import cv2

        source = Path(video_path).resolve()
        context_frames = max(0, int(round(self.context_seconds * max(float(fps), 1.0))))
        evidence_start = max(
            event.start_frame,
            event.transition_start_frame - context_frames,
        )
        evidence_end = min(
            event.end_frame,
            event.transition_end_frame + context_frames,
        )
        event_dir = (
            Path(output_root).resolve()
            / source.stem
            / (
                f"person_{event.person_id}_transition_"
                f"{event.transition_start_frame}_{event.transition_end_frame}"
            )
        )
        event_dir.mkdir(parents=True, exist_ok=True)

        target_track = sorted(
            [
                item
                for item in observations
                if item.class_id == 0
                and item.track_id == event.person_id
                and evidence_start <= item.frame_index <= evidence_end
            ],
            key=lambda item: item.frame_index,
        )
        if not target_track:
            raise RuntimeError(f"No observations found for person {event.person_id}")

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for evidence generation: {source}")

        generated: list[EvidenceImage] = []
        try:
            for position in self.sample_positions:
                requested = int(round(evidence_start + position * (evidence_end - evidence_start)))
                target = min(target_track, key=lambda item: abs(item.frame_index - requested))
                capture.set(cv2.CAP_PROP_POS_FRAMES, target.frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not decode frame {target.frame_index} from {source}")

                height, width = frame.shape[:2]
                x1 = max(0, min(width - 1, int(round(target.box.x1 * width))))
                y1 = max(0, min(height - 1, int(round(target.box.y1 * height))))
                x2 = max(x1 + 1, min(width, int(round(target.box.x2 * width))))
                y2 = max(y1 + 1, min(height, int(round(target.box.y2 * height))))

                context = frame.copy()
                self._draw_target(context, x1, y1, x2, y2, event.person_id)
                context = self._resize(context)

                box_width = x2 - x1
                box_height = y2 - y1
                margin_x = int(round(box_width * self.crop_margin))
                margin_y = int(round(box_height * self.crop_margin))
                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(width, x2 + margin_x)
                crop_y2 = min(height, y2 + margin_y)
                focus = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                self._draw_target(
                    focus,
                    x1 - crop_x1,
                    y1 - crop_y1,
                    x2 - crop_x1,
                    y2 - crop_y1,
                    event.person_id,
                )
                focus = self._resize(focus)

                context_path = event_dir / f"frame_{target.frame_index:06d}_context.jpg"
                focus_path = event_dir / f"frame_{target.frame_index:06d}_focus.jpg"
                parameters = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                if not cv2.imwrite(str(context_path), context, parameters):
                    raise RuntimeError(f"Could not save evidence image: {context_path}")
                if not cv2.imwrite(str(focus_path), focus, parameters):
                    raise RuntimeError(f"Could not save evidence image: {focus_path}")

                generated.append(
                    EvidenceImage(
                        frame_index=target.frame_index,
                        context_path=context_path,
                        focus_path=focus_path,
                    )
                )
        finally:
            capture.release()

        return generated

    @staticmethod
    def _draw_target(image, x1: int, y1: int, x2: int, y2: int, person_id: int) -> None:
        import cv2

        colour = (0, 0, 255)
        thickness = max(2, int(round(max(image.shape[:2]) / 400)))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
        label = f"TARGET PERSON {person_id}"
        label_y = max(25, y1 - 8)
        cv2.putText(
            image,
            label,
            (max(0, x1), label_y),
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
