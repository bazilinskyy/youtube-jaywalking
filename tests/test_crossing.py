"""Synthetic tests for crossing detection and false crossing filters."""

import unittest

from crowd_jaywalking.crossing import CrossingDetector
from crowd_jaywalking.models import BoundingBox, RejectionReason, TrackObservation


def settings() -> dict:
    return {
        "road_left": 0.45,
        "road_right": 0.55,
        "boundary_tolerance": 0.0,
        "min_track_seconds": 0.30,
        "min_road_seconds": 0.10,
        "max_track_gap_seconds": 1.0,
        "min_crossing_x_range": 0.14,
        "low_x_range": 0.30,
        "low_x_min_road_seconds": 0.20,
        "weak_x_range": 0.64,
        "long_weak_road_seconds": 3.0,
        "weak_y_jitter_x_range": 0.50,
        "weak_y_jitter_motion": 0.30,
        "weak_y_jitter_height": 0.22,
        "tiny_track_height": 0.12,
        "tiny_track_width": 0.026,
        "tiny_track_min_road_seconds": 0.33,
        "min_static_shared_seconds": 0.27,
        "camera_ratio_threshold": 0.60,
        "min_relative_x_range": 0.01,
        "rider_min_shared_seconds": 0.20,
        "rider_overlap_threshold": 0.35,
    }


def observation(
    frame: int,
    x: float,
    *,
    track_id: int = 1,
    class_id: int = 0,
    width: float = 0.08,
    height: float = 0.20,
) -> TrackObservation:
    return TrackObservation(
        frame_index=frame,
        track_id=track_id,
        class_id=class_id,
        confidence=0.95,
        box=BoundingBox(
            x1=x - width / 2,
            y1=0.50 - height / 2,
            x2=x + width / 2,
            y2=0.50 + height / 2,
        ),
    )


def crossing_track() -> list[TrackObservation]:
    return [
        observation(0, 0.30),
        observation(1, 0.46),
        observation(2, 0.50),
        observation(3, 0.54),
        observation(4, 0.70),
    ]


class CrossingDetectorTests(unittest.TestCase):
    def test_accepts_complete_crossing(self) -> None:
        result = CrossingDetector(settings()).detect(crossing_track(), fps=10.0)
        self.assertEqual(len(result.valid_events), 1)
        self.assertEqual(result.valid_events[0].person_id, 1)
        self.assertEqual(result.valid_events[0].rejection_reason, RejectionReason.NONE)

    def test_validates_full_segment_not_narrow_transition_slice(self) -> None:
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.20, 0.35, 0.44, 0.46, 0.50, 0.54, 0.56, 0.65, 0.80])
        ]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertEqual(len(result.valid_events), 1)
        event = result.valid_events[0]
        self.assertAlmostEqual(event.features.x_range, 0.60)
        self.assertEqual((event.start_frame, event.end_frame), (0, 8))
        self.assertEqual((event.transition_start_frame, event.transition_end_frame), (2, 6))

    def test_rejects_insufficient_lateral_motion(self) -> None:
        track = [
            observation(0, 0.44),
            observation(1, 0.47),
            observation(2, 0.50),
            observation(3, 0.53),
            observation(4, 0.56),
        ]
        result = CrossingDetector(settings()).detect(track, fps=10.0)
        self.assertFalse(result.valid_events)
        self.assertEqual(
            result.rejected_events[0].rejection_reason,
            RejectionReason.INSUFFICIENT_LATERAL_MOTION,
        )

    def test_rejects_camera_motion(self) -> None:
        people = crossing_track()
        static_objects = [
            observation(
                item.frame_index,
                item.box.centre_x,
                track_id=99,
                class_id=9,
                width=0.05,
                height=0.10,
            )
            for item in people
        ]
        result = CrossingDetector(settings()).detect(people + static_objects, fps=10.0)
        self.assertFalse(result.valid_events)
        self.assertEqual(
            result.rejected_events[0].rejection_reason,
            RejectionReason.CAMERA_MOTION,
        )

    def test_rejects_rider(self) -> None:
        people = crossing_track()
        vehicles = [
            observation(
                item.frame_index,
                item.box.centre_x,
                track_id=50,
                class_id=1,
                width=0.12,
                height=0.24,
            )
            for item in people
        ]
        result = CrossingDetector(settings()).detect(people + vehicles, fps=10.0)
        self.assertFalse(result.valid_events)
        self.assertEqual(result.rejected_events[0].rejection_reason, RejectionReason.RIDER)


if __name__ == "__main__":
    unittest.main()
