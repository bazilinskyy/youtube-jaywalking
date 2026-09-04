"""CROWD City person crossing detection with structured diagnostics.

The rules follow crowd-dataset/crowd-city commit
205ebcdb5f2cb994db76dcd0c4471ceb72554f42.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from statistics import median
from typing import Any, Iterable

from .models import CrossingEvent, CrossingFeatures, RejectionReason, TrackObservation

PERSON_CLASS = 0
RIDER_CLASSES = {1, 3}
STATIC_CLASSES = {9, 10, 11, 12, 13}
EPS = 1e-9


@dataclass(frozen=True)
class CrossingDetectionResult:
    valid_events: list[CrossingEvent]
    rejected_events: list[CrossingEvent]


def _q(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[max(0, min(round((len(ordered) - 1) * probability), len(ordered) - 1))]


def _range(values: Iterable[float]) -> float:
    materialised = list(values)
    return max(0.0, _q(materialised, 0.95) - _q(materialised, 0.05)) if materialised else 0.0


def _frames(seconds: float, fps: float, minimum: int = 1) -> int:
    return max(minimum, round(float(seconds) * max(float(fps), 1.0)))


def _longest_run(frames: Iterable[int], gap_allow: int) -> int:
    ordered = sorted({int(frame) for frame in frames})
    if not ordered:
        return 0
    longest = current = 1
    for previous, current_frame in zip(ordered, ordered[1:]):
        if current_frame - previous <= max(gap_allow, 0) + 1:
            current += 1
        else:
            longest, current = max(longest, current), 1
    return max(longest, current)


class CrossingDetector:
    """Apply complete and conservative partial CROWD crossing rules."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.left = float(settings["road_left"])
        self.right = float(settings["road_right"])
        self.tolerance = float(settings.get("boundary_tolerance", 0.0))

    def f(self, key: str, fps: float) -> int:
        return _frames(self.settings[key], fps)

    def detect(self, observations: list[TrackObservation], fps: float) -> CrossingDetectionResult:
        data = self._deduplicate(observations)
        people: dict[int, list[TrackObservation]] = defaultdict(list)
        for item in data:
            if item.class_id == PERSON_CLASS:
                people[item.track_id].append(item)
        accepted: list[CrossingEvent] = []
        rejected: list[CrossingEvent] = []
        for person_id, track in people.items():
            track.sort(key=lambda item: item.frame_index)
            for segment in self._split(track, self.f("max_track_gap_seconds", fps)):
                if len(segment) < self.f("min_track_seconds", fps):
                    continue
                min_road = self.f("min_road_seconds", fps)
                window = self._window(segment, min_road)
                partial = False
                if window is None and bool(self.settings.get("partial_crossing_enabled", True)):
                    window = self._partial_window(segment, min_road)
                    partial = window is not None
                if window is None:
                    continue
                event = self._event(person_id, segment, window, data, fps, partial=partial)
                (accepted if event.valid else rejected).append(event)
        accepted.sort(key=lambda item: (item.start_frame, item.person_id))
        rejected.sort(key=lambda item: (item.start_frame, item.person_id))
        return CrossingDetectionResult(accepted, rejected)

    @staticmethod
    def _deduplicate(items: list[TrackObservation]) -> list[TrackObservation]:
        best: dict[tuple[int, int, int], TrackObservation] = {}
        for item in items:
            key = item.class_id, item.track_id, item.frame_index
            if key not in best or item.confidence > best[key].confidence:
                best[key] = item
        return sorted(best.values(), key=lambda item: (item.frame_index, item.class_id, item.track_id))

    @staticmethod
    def _split(track: list[TrackObservation], max_gap: int) -> list[list[TrackObservation]]:
        if not track:
            return []
        parts = [[track[0]]]
        for item in track[1:]:
            if item.frame_index - parts[-1][-1].frame_index > max_gap:
                parts.append([item])
            else:
                parts[-1].append(item)
        return parts

    def _states(self, track: list[TrackObservation]) -> list[str]:
        states: list[str] = []
        previous = "ROAD"
        for index, item in enumerate(track):
            x = item.box.centre_x
            if index == 0:
                previous = "LEFT" if x < self.left else "RIGHT" if x > self.right else "ROAD"
            elif x <= self.left - self.tolerance:
                previous = "LEFT"
            elif x >= self.right + self.tolerance:
                previous = "RIGHT"
            elif self.left + self.tolerance <= x <= self.right - self.tolerance:
                previous = "ROAD"
            states.append(previous)
        return states

    def _window(self, track: list[TrackObservation], min_road: int) -> tuple[int, int] | None:
        states = self._states(track)
        if states.count("ROAD") < min_road:
            return None
        for road_index, state in enumerate(states):
            if state != "ROAD":
                continue
            before = states[:road_index]
            after = states[road_index + 1 :]
            if "LEFT" in before and "RIGHT" in after:
                return len(before) - 1 - before[::-1].index("LEFT"), road_index + 1 + after.index("RIGHT")
            if "RIGHT" in before and "LEFT" in after:
                return len(before) - 1 - before[::-1].index("RIGHT"), road_index + 1 + after.index("LEFT")
        return None

    def _partial_window(
        self,
        track: list[TrackObservation],
        min_road: int,
    ) -> tuple[int, int] | None:
        """Find a train-tuned partial crossing at a video boundary.

        The JAAD train diagnostics support two symmetric, deliberately narrow
        cases: a side-to-road entry that ends while the person remains on the
        road, and a road-to-side exit that starts after the person was already
        on the road. A partial exit additionally needs a large lateral range;
        this separated the train positives from non-crossings without adding a
        train false positive.
        """

        states = self._states(track)
        if not states:
            return None

        start_state = states[0]
        end_state = states[-1]

        if start_state in {"LEFT", "RIGHT"} and end_state == "ROAD":
            road_start = len(states) - 1
            while road_start > 0 and states[road_start - 1] == "ROAD":
                road_start -= 1
            if len(states) - road_start >= min_road:
                return max(0, road_start - 1), len(states) - 1

        if start_state == "ROAD" and end_state in {"LEFT", "RIGHT"}:
            road_end = 0
            while road_end + 1 < len(states) and states[road_end + 1] == "ROAD":
                road_end += 1
            x_values = [item.box.centre_x for item in track]
            x_range = max(x_values) - min(x_values)
            if (
                road_end + 1 >= min_road
                and x_range >= float(self.settings.get("partial_exit_min_x_range", 0.48))
            ):
                return 0, min(len(states) - 1, road_end + 1)

        return None

    def _event(
        self,
        person_id: int,
        track: list[TrackObservation],
        window: tuple[int, int],
        all_items: list[TrackObservation],
        fps: float,
        *,
        partial: bool = False,
    ) -> CrossingEvent:
        xs = [item.box.centre_x for item in track]
        ys = [item.box.centre_y for item in track]
        static = self._static_stats(track, all_items)
        span = max(xs) - min(xs)
        features = CrossingFeatures(
            track_frames=len(track),
            road_frames=self._states(track).count("ROAD"),
            x_range=span,
            x_speed_per_frame=span / max(1, track[-1].frame_index - track[0].frame_index + 1),
            y_gross_motion=sum(abs(b - a) for a, b in zip(ys, ys[1:])),
            median_width=float(median(item.box.width for item in track)),
            median_height=float(median(item.box.height for item in track)),
            static_shared_frames=int(static["shared"]),
            static_x_range=float(static["sx"]),
            relative_x_range=float(static["relx"]),
            camera_motion_ratio=float(static["ratio"]),
        )
        reason = (
            self._partial_reason(track, all_items, features, fps)
            if partial
            else self._reason(track, all_items, features, fps)
        )
        return CrossingEvent(
            person_id=person_id,
            start_frame=track[0].frame_index,
            end_frame=track[-1].frame_index,
            transition_start_frame=track[window[0]].frame_index,
            transition_end_frame=track[window[1]].frame_index,
            valid=reason == RejectionReason.NONE,
            rejection_reason=reason,
            features=features,
        )

    def _reason(
        self,
        track: list[TrackObservation],
        all_items: list[TrackObservation],
        x: CrossingFeatures,
        fps: float,
    ) -> RejectionReason:
        s = self.settings
        if self._is_rider(track, all_items, fps):
            return RejectionReason.RIDER
        if x.x_range < s["min_crossing_x_range"]:
            return RejectionReason.INSUFFICIENT_LATERAL_MOTION
        if x.x_range < s["low_x_range"] and x.road_frames < self.f("low_x_min_road_seconds", fps):
            return RejectionReason.INSUFFICIENT_LATERAL_MOTION
        if x.x_range < s["weak_x_range"] and x.road_frames > self.f("long_weak_road_seconds", fps):
            return RejectionReason.LONG_WEAK_TRACK
        if (
            x.x_range < s["large_lateral_x_range"]
            and x.road_frames > self.f("jitter_road_seconds", fps)
            and x.y_gross_motion > s["weak_y_jitter_motion"]
        ):
            return RejectionReason.VERTICAL_JITTER
        if (
            x.x_range < s["weak_y_jitter_x_range"]
            and x.y_gross_motion > s["weak_y_jitter_motion"]
            and x.median_height < s["weak_y_jitter_height"]
        ):
            return RejectionReason.VERTICAL_JITTER
        if (
            x.x_range < s["tiny_long_track_x_range"]
            and x.median_height < s["tiny_long_track_height"]
            and x.road_frames >= self.f("tiny_long_track_road_seconds", fps)
        ):
            return RejectionReason.TINY_UNVERIFIED_TRACK

        min_static = self.f("min_static_shared_seconds", fps)
        if (
            x.static_shared_frames < min_static
            and x.median_height <= s["tiny_no_static_height"]
            and x.median_width <= s["tiny_no_static_width"]
            and (
                x.road_frames >= self.f("tiny_no_static_min_road_seconds", fps)
                or x.road_frames >= self.f("no_static_tiny_min_road_seconds", fps)
                or x.x_speed_per_frame >= s["no_static_tiny_fast_speed"]
            )
        ):
            return RejectionReason.TINY_UNVERIFIED_TRACK

        camera_dominant = (
            x.static_shared_frames >= min_static
            and x.static_x_range >= s["camera_static_x_range"]
            and x.camera_motion_ratio >= s["camera_ratio_threshold"]
        )
        if camera_dominant and (
            (x.median_height <= s["camera_tiny_height"] and x.road_frames >= self.f("camera_min_road_seconds", fps))
            or (x.relative_x_range <= s["camera_static_tiny_relative_x_range"] and x.median_height <= s["camera_static_tiny_height"])
            or (x.relative_x_range <= s["camera_static_relative_x_range"] and x.median_height <= s["camera_static_height"])
        ):
            return RejectionReason.CAMERA_MOTION

        slender = (
            x.median_width <= s["slender_track_width"]
            and x.median_height < s["slender_track_height"]
            and self.f("slender_track_min_road_seconds", fps)
            <= x.road_frames
            <= self.f("slender_track_max_road_seconds", fps)
        )
        if slender:
            if x.static_shared_frames < min_static:
                if x.median_height < s["no_static_slender_height"] and x.road_frames <= self.f("no_static_slender_max_road_seconds", fps):
                    return RejectionReason.TINY_UNVERIFIED_TRACK
            elif x.relative_x_range < s["slender_static_min_relative_x_range"] or (
                camera_dominant and x.median_height <= s["camera_tiny_height"]
            ):
                return RejectionReason.CAMERA_MOTION

        if (
            x.x_range > s["large_lateral_x_range"]
            and x.median_height < s["large_lateral_tiny_height"]
            and x.road_frames >= self.f("camera_min_road_seconds", fps)
            and camera_dominant
            and x.relative_x_range < 0.20
        ):
            return RejectionReason.CAMERA_MOTION
        maximum_speed = s.get("max_crossing_speed_per_frame")
        if maximum_speed is not None and x.x_speed_per_frame > maximum_speed:
            return RejectionReason.INSUFFICIENT_LATERAL_MOTION
        if x.static_shared_frames >= min_static:
            if x.relative_x_range < s["min_relative_x_range"]:
                return RejectionReason.CAMERA_MOTION
            if x.camera_motion_ratio >= s["camera_ratio_threshold"] and x.relative_x_range < 2 * s["min_relative_x_range"]:
                return RejectionReason.CAMERA_MOTION
        return RejectionReason.NONE

    def _partial_reason(
        self,
        track: list[TrackObservation],
        all_items: list[TrackObservation],
        x: CrossingFeatures,
        fps: float,
    ) -> RejectionReason:
        """Retain high-value false-positive filters for partial crossings.

        Motion and size filters used for complete traversals cannot be applied
        unchanged to boundary-truncated tracks. Rider and camera-motion checks
        remain valid because they use co-motion evidence rather than requiring
        the whole crossing to be visible.
        """

        if self._is_rider(track, all_items, fps):
            return RejectionReason.RIDER
        if self._is_camera_motion(x, fps):
            return RejectionReason.CAMERA_MOTION
        return RejectionReason.NONE

    def _is_camera_motion(self, x: CrossingFeatures, fps: float) -> bool:
        s = self.settings
        min_static = self.f("min_static_shared_seconds", fps)
        if x.static_shared_frames < min_static:
            return False

        camera_dominant = (
            x.static_x_range >= s["camera_static_x_range"]
            and x.camera_motion_ratio >= s["camera_ratio_threshold"]
        )
        if camera_dominant and (
            (
                x.median_height <= s["camera_tiny_height"]
                and x.road_frames >= self.f("camera_min_road_seconds", fps)
            )
            or (
                x.relative_x_range <= s["camera_static_tiny_relative_x_range"]
                and x.median_height <= s["camera_static_tiny_height"]
            )
            or (
                x.relative_x_range <= s["camera_static_relative_x_range"]
                and x.median_height <= s["camera_static_height"]
            )
        ):
            return True

        slender = (
            x.median_width <= s["slender_track_width"]
            and x.median_height < s["slender_track_height"]
            and self.f("slender_track_min_road_seconds", fps)
            <= x.road_frames
            <= self.f("slender_track_max_road_seconds", fps)
        )
        if slender and (
            x.relative_x_range < s["slender_static_min_relative_x_range"]
            or (camera_dominant and x.median_height <= s["camera_tiny_height"])
        ):
            return True

        if (
            x.x_range > s["large_lateral_x_range"]
            and x.median_height < s["large_lateral_tiny_height"]
            and x.road_frames >= self.f("camera_min_road_seconds", fps)
            and camera_dominant
            and x.relative_x_range < 0.20
        ):
            return True

        return (
            x.relative_x_range < s["min_relative_x_range"]
            or (
                x.camera_motion_ratio >= s["camera_ratio_threshold"]
                and x.relative_x_range < 2 * s["min_relative_x_range"]
            )
        )

    def _static_stats(
        self,
        track: list[TrackObservation],
        all_items: list[TrackObservation],
    ) -> dict[str, float | int]:
        people = {item.frame_index: item for item in track}
        static: dict[int, dict[int, TrackObservation]] = defaultdict(dict)
        for item in all_items:
            if track[0].frame_index <= item.frame_index <= track[-1].frame_index and item.class_id in STATIC_CLASSES:
                static[item.track_id][item.frame_index] = item
        best: dict[str, float | int] | None = None
        for reference in static.values():
            shared = sorted(set(people).intersection(reference))
            if not shared:
                continue
            px = [people[frame].box.centre_x for frame in shared]
            sx = [reference[frame].box.centre_x for frame in shared]
            candidate = {
                "shared": len(shared),
                "sx": _range(sx),
                "relx": _range(a - b for a, b in zip(px, sx)),
                "ratio": _range(sx) / max(_range(px), EPS),
            }
            if best is None or (candidate["shared"], candidate["sx"]) > (best["shared"], best["sx"]):
                best = candidate
        return best or {"shared": 0, "sx": 0.0, "relx": 0.0, "ratio": 0.0}

    def _is_rider(
        self,
        track: list[TrackObservation],
        all_items: list[TrackObservation],
        fps: float,
    ) -> bool:
        s = self.settings
        people = {item.frame_index: item for item in track}
        vehicles: dict[int, dict[int, TrackObservation]] = defaultdict(dict)
        for item in all_items:
            if item.class_id in RIDER_CLASSES:
                vehicles[item.track_id][item.frame_index] = item
        for vehicle in vehicles.values():
            shared = sorted(set(people).intersection(vehicle))
            if len(shared) < self.f("rider_min_shared_seconds", fps):
                continue
            gap = max(0, round(s["rider_shared_run_gap_seconds"] * fps))
            if _longest_run(shared, gap) < self.f("rider_min_continuous_shared_seconds", fps):
                continue
            p = [people[frame] for frame in shared]
            v = [vehicle[frame] for frame in shared]
            width_pass = sum(
                vb.box.width / max(pb.box.width, EPS) >= s["rider_min_vehicle_width_ratio"]
                for pb, vb in zip(p, v)
            ) / len(shared)
            if width_pass < s["rider_min_vehicle_width_ratio_frames"]:
                continue
            proximity: list[bool] = []
            spatial: list[bool] = []
            for pb, vb in zip(p, v):
                dx = vb.box.centre_x - pb.box.centre_x
                dy = vb.box.centre_y - pb.box.centre_y
                proximity.append(hypot(dx, dy) / max(pb.box.height, EPS) < s["rider_distance_relative_threshold"])
                spatial.append(
                    abs(dx) < s["rider_alpha_x"] * pb.box.width
                    and s["rider_beta_y"] * pb.box.height < dy < s["rider_gamma_y"] * pb.box.height
                )
            if sum(proximity) / len(shared) < s["rider_proximity_ratio"]:
                continue
            coloc = sum(a and b for a, b in zip(proximity, spatial)) / len(shared)
            similar = 0
            moving = 0
            for index in range(1, len(shared)):
                pdx = p[index].box.centre_x - p[index - 1].box.centre_x
                pdy = p[index].box.centre_y - p[index - 1].box.centre_y
                vdx = v[index].box.centre_x - v[index - 1].box.centre_x
                vdy = v[index].box.centre_y - v[index - 1].box.centre_y
                pn, vn = hypot(pdx, pdy), hypot(vdx, vdy)
                if proximity[index] and pn > EPS and vn > EPS:
                    moving += 1
                    similar += (pdx * vdx + pdy * vdy) / (pn * vn) > s["rider_similarity_threshold"]
            sim = similar / moving if moving >= self.f("rider_min_motion_seconds", fps) else 0.0
            if len(shared) < self.f("rider_short_shared_seconds", fps):
                displacement = hypot(
                    p[-1].box.centre_x - p[0].box.centre_x,
                    p[-1].box.centre_y - p[0].box.centre_y,
                ) / max(sum(item.box.height for item in p) / len(p), EPS)
                if sim < s["rider_short_similarity_ratio"] and displacement < s["rider_short_displacement"]:
                    continue
            if coloc >= s["rider_colocation_ratio"] or (
                sim >= s["rider_similarity_ratio"] and coloc >= s["rider_motion_colocation_min"]
            ):
                return True
        return False
