"""Person-specific road crossing detection and false crossing filters.

The state transition and filter design is adapted from the CROWD-city
crossing detector, with explicit inputs and structured rejection reasons.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable

from .models import CrossingEvent, CrossingFeatures, RejectionReason, TrackObservation


PERSON_CLASS = 0
RIDER_VEHICLE_CLASSES = {1, 2, 3, 5, 7}
STATIC_REFERENCE_CLASSES = {9, 10, 11, 12, 13}


@dataclass(frozen=True)
class CrossingDetectionResult:
    """Accepted and rejected crossing candidates."""

    valid_events: list[CrossingEvent]
    rejected_events: list[CrossingEvent]


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _robust_range(values: Iterable[float], tail: float = 0.05) -> float:
    materialised = list(values)
    if not materialised:
        return 0.0
    return max(0.0, _quantile(materialised, 1.0 - tail) - _quantile(materialised, tail))


def _seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    return max(minimum, int(round(float(seconds) * max(float(fps), 1.0))))


class CrossingDetector:
    """Detect complete side to road to opposite-side person transitions."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.left = float(settings["road_left"])
        self.right = float(settings["road_right"])
        self.tolerance = float(settings.get("boundary_tolerance", 0.0))

    def detect(self, observations: list[TrackObservation], fps: float) -> CrossingDetectionResult:
        """Detect and validate crossing events for every tracked person."""

        deduplicated = self._deduplicate(observations)
        person_tracks: dict[int, list[TrackObservation]] = defaultdict(list)
        for observation in deduplicated:
            if observation.class_id == PERSON_CLASS:
                person_tracks[observation.track_id].append(observation)

        max_gap = _seconds_to_frames(
            float(self.settings["max_track_gap_seconds"]),
            fps,
            minimum=1,
        )
        min_track = _seconds_to_frames(float(self.settings["min_track_seconds"]), fps)
        min_road = _seconds_to_frames(float(self.settings["min_road_seconds"]), fps)

        valid: list[CrossingEvent] = []
        rejected: list[CrossingEvent] = []

        for person_id, track in person_tracks.items():
            track.sort(key=lambda item: item.frame_index)
            for segment in self._split_track(track, max_gap):
                if len(segment) < min_track:
                    continue

                window = self._find_crossing_window(segment, min_road)
                if window is None:
                    continue

                # The corridor transition locates the crossing, but the complete
                # continuous segment supplies motion features. Using only the
                # transition slice constrains x_range to roughly road_right -
                # road_left and contradicts a larger min_crossing_x_range.
                event = self._validate_candidate(
                    person_id=person_id,
                    person_track=segment,
                    transition_start_frame=segment[window[0]].frame_index,
                    transition_end_frame=segment[window[1]].frame_index,
                    all_observations=deduplicated,
                    fps=fps,
                )
                if event.valid:
                    valid.append(event)
                else:
                    rejected.append(event)

        valid.sort(key=lambda item: (item.start_frame, item.person_id))
        rejected.sort(key=lambda item: (item.start_frame, item.person_id))
        return CrossingDetectionResult(valid_events=valid, rejected_events=rejected)

    @staticmethod
    def _deduplicate(observations: list[TrackObservation]) -> list[TrackObservation]:
        best: dict[tuple[int, int, int], TrackObservation] = {}
        for observation in observations:
            key = (observation.class_id, observation.track_id, observation.frame_index)
            previous = best.get(key)
            if previous is None or observation.confidence > previous.confidence:
                best[key] = observation
        return sorted(best.values(), key=lambda item: (item.frame_index, item.class_id, item.track_id))

    @staticmethod
    def _split_track(track: list[TrackObservation], max_gap: int) -> list[list[TrackObservation]]:
        if not track:
            return []
        segments: list[list[TrackObservation]] = [[track[0]]]
        for observation in track[1:]:
            if observation.frame_index - segments[-1][-1].frame_index > max_gap:
                segments.append([observation])
            else:
                segments[-1].append(observation)
        return segments

    def _states(self, track: list[TrackObservation]) -> list[str]:
        left_hard = self.left - self.tolerance
        left_soft = self.left + self.tolerance
        right_soft = self.right - self.tolerance
        right_hard = self.right + self.tolerance
        states: list[str] = []
        previous = "ROAD"

        for index, observation in enumerate(track):
            x = observation.box.centre_x
            if index == 0:
                if x < self.left:
                    previous = "LEFT"
                elif x > self.right:
                    previous = "RIGHT"
                else:
                    previous = "ROAD"
            elif x <= left_hard:
                previous = "LEFT"
            elif x >= right_hard:
                previous = "RIGHT"
            elif left_soft <= x <= right_soft:
                previous = "ROAD"
            states.append(previous)
        return states

    def _find_crossing_window(
        self,
        track: list[TrackObservation],
        min_road_frames: int,
    ) -> tuple[int, int] | None:
        states = self._states(track)
        if states.count("ROAD") < min_road_frames:
            return None

        for road_index, state in enumerate(states):
            if state != "ROAD":
                continue

            left_before = [index for index in range(road_index) if states[index] == "LEFT"]
            right_before = [index for index in range(road_index) if states[index] == "RIGHT"]
            left_after = [index for index in range(road_index + 1, len(states)) if states[index] == "LEFT"]
            right_after = [index for index in range(road_index + 1, len(states)) if states[index] == "RIGHT"]

            if left_before and right_after:
                return left_before[-1], right_after[0]
            if right_before and left_after:
                return right_before[-1], left_after[0]

        return None

    def _validate_candidate(
        self,
        person_id: int,
        person_track: list[TrackObservation],
        transition_start_frame: int,
        transition_end_frame: int,
        all_observations: list[TrackObservation],
        fps: float,
    ) -> CrossingEvent:
        frames = [item.frame_index for item in person_track]
        xs = [item.box.centre_x for item in person_track]
        ys = [item.box.centre_y for item in person_track]
        widths = [item.box.width for item in person_track]
        heights = [item.box.height for item in person_track]
        states = self._states(person_track)

        duration_frames = max(1, frames[-1] - frames[0] + 1)
        x_range = max(xs) - min(xs)
        x_speed = x_range / duration_frames
        road_frames = states.count("ROAD")
        y_motion = sum(abs(current - previous) for previous, current in zip(ys, ys[1:]))
        median_width = float(median(widths))
        median_height = float(median(heights))

        static = self._static_motion(person_track, all_observations)
        features = CrossingFeatures(
            track_frames=len(person_track),
            road_frames=road_frames,
            x_range=x_range,
            x_speed_per_frame=x_speed,
            y_gross_motion=y_motion,
            median_width=median_width,
            median_height=median_height,
            static_shared_frames=static["shared_frames"],
            static_x_range=static["static_x_range"],
            relative_x_range=static["relative_x_range"],
            camera_motion_ratio=static["camera_ratio"],
        )

        reason = self._rejection_reason(
            person_track=person_track,
            all_observations=all_observations,
            features=features,
            fps=fps,
        )
        return CrossingEvent(
            person_id=person_id,
            start_frame=frames[0],
            end_frame=frames[-1],
            transition_start_frame=transition_start_frame,
            transition_end_frame=transition_end_frame,
            valid=reason == RejectionReason.NONE,
            rejection_reason=reason,
            features=features,
        )

    def _rejection_reason(
        self,
        person_track: list[TrackObservation],
        all_observations: list[TrackObservation],
        features: CrossingFeatures,
        fps: float,
    ) -> RejectionReason:
        if self._is_rider(person_track, all_observations, fps):
            return RejectionReason.RIDER

        if features.x_range < float(self.settings["min_crossing_x_range"]):
            return RejectionReason.INSUFFICIENT_LATERAL_MOTION

        low_road_frames = _seconds_to_frames(float(self.settings["low_x_min_road_seconds"]), fps)
        if features.x_range < float(self.settings["low_x_range"]) and features.road_frames < low_road_frames:
            return RejectionReason.INSUFFICIENT_LATERAL_MOTION

        long_road_frames = _seconds_to_frames(float(self.settings["long_weak_road_seconds"]), fps)
        if features.x_range < float(self.settings["weak_x_range"]) and features.road_frames > long_road_frames:
            return RejectionReason.LONG_WEAK_TRACK

        if (
            features.x_range < float(self.settings["weak_y_jitter_x_range"])
            and features.y_gross_motion > float(self.settings["weak_y_jitter_motion"])
            and features.median_height < float(self.settings["weak_y_jitter_height"])
        ):
            return RejectionReason.VERTICAL_JITTER

        min_static = _seconds_to_frames(float(self.settings["min_static_shared_seconds"]), fps)
        tiny_road = _seconds_to_frames(float(self.settings["tiny_track_min_road_seconds"]), fps)
        if (
            features.static_shared_frames < min_static
            and features.median_height <= float(self.settings["tiny_track_height"])
            and features.median_width <= float(self.settings["tiny_track_width"])
            and features.road_frames >= tiny_road
        ):
            return RejectionReason.TINY_UNVERIFIED_TRACK

        if features.static_shared_frames >= min_static:
            relative_limit = float(self.settings["min_relative_x_range"])
            ratio_limit = float(self.settings["camera_ratio_threshold"])
            if features.relative_x_range < relative_limit:
                return RejectionReason.CAMERA_MOTION
            if features.camera_motion_ratio >= ratio_limit and features.relative_x_range < 2.0 * relative_limit:
                return RejectionReason.CAMERA_MOTION

        return RejectionReason.NONE

    def _static_motion(
        self,
        person_track: list[TrackObservation],
        all_observations: list[TrackObservation],
    ) -> dict[str, float | int]:
        person_by_frame = {item.frame_index: item for item in person_track}
        start = person_track[0].frame_index
        end = person_track[-1].frame_index
        static_tracks: dict[int, list[TrackObservation]] = defaultdict(list)

        for observation in all_observations:
            if start <= observation.frame_index <= end and observation.class_id in STATIC_REFERENCE_CLASSES:
                static_tracks[observation.track_id].append(observation)

        best: dict[str, float | int] | None = None
        for static_track in static_tracks.values():
            static_by_frame = {item.frame_index: item for item in static_track}
            shared_frames = sorted(set(person_by_frame).intersection(static_by_frame))
            if not shared_frames:
                continue

            person_x = [person_by_frame[frame].box.centre_x for frame in shared_frames]
            static_x = [static_by_frame[frame].box.centre_x for frame in shared_frames]
            person_x_range = _robust_range(person_x)
            static_x_range = _robust_range(static_x)
            relative_x_range = _robust_range(
                person_value - static_value for person_value, static_value in zip(person_x, static_x)
            )
            camera_ratio = static_x_range / max(person_x_range, 1e-9)
            candidate: dict[str, float | int] = {
                "shared_frames": len(shared_frames),
                "static_x_range": static_x_range,
                "relative_x_range": relative_x_range,
                "camera_ratio": camera_ratio,
            }
            if best is None or (
                int(candidate["shared_frames"]), float(candidate["static_x_range"])
            ) > (
                int(best["shared_frames"]), float(best["static_x_range"])
            ):
                best = candidate

        return best or {
            "shared_frames": 0,
            "static_x_range": 0.0,
            "relative_x_range": 0.0,
            "camera_ratio": 0.0,
        }

    def _is_rider(
        self,
        person_track: list[TrackObservation],
        all_observations: list[TrackObservation],
        fps: float,
    ) -> bool:
        minimum_shared = _seconds_to_frames(float(self.settings["rider_min_shared_seconds"]), fps)
        overlap_threshold = float(self.settings["rider_overlap_threshold"])
        vehicles_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
        for observation in all_observations:
            if observation.class_id in RIDER_VEHICLE_CLASSES:
                vehicles_by_frame[observation.frame_index].append(observation)

        shared = 0
        for person in person_track:
            person_area = max(person.box.width * person.box.height, 1e-9)
            for vehicle in vehicles_by_frame.get(person.frame_index, []):
                intersection_x = max(
                    0.0,
                    min(person.box.x2, vehicle.box.x2) - max(person.box.x1, vehicle.box.x1),
                )
                intersection_y = max(
                    0.0,
                    min(person.box.y2, vehicle.box.y2) - max(person.box.y1, vehicle.box.y1),
                )
                overlap = (intersection_x * intersection_y) / person_area
                if overlap >= overlap_threshold:
                    shared += 1
                    break

        return shared >= minimum_shared
