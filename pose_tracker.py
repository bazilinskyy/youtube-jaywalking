from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from pose_estimator import Pose, PoseEstimator

_CROSSING_WINDOW = 10  # last N poses used by crossing_intent


@dataclass
class PoseTrack:
    track_id: int
    poses: list[Pose] = field(default_factory=list)

    def latest_pose(self) -> Pose | None:
        return self.poses[-1] if self.poses else None

    def crossing_intent(self, estimator: PoseEstimator, n: int = _CROSSING_WINDOW) -> float:
        """Fraction of last n poses where is_crossing=True (0.0–1.0)."""
        recent = self.poses[-n:]
        if not recent:
            return 0.0
        return sum(estimator.is_crossing(p) for p in recent) / len(recent)


class PoseTracker:
    def __init__(self, estimator: PoseEstimator, max_history: int = 30):
        self._estimator = estimator
        self._max_history = max_history
        self._tracks: dict[int, PoseTrack] = {}

    def update(
        self,
        frame: np.ndarray,
        track_id_boxes: list[tuple],
        frame_num: int,
    ) -> dict[int, PoseTrack]:
        """
        Estimate poses for the current frame and append to each track's history.

        Args:
            frame: BGR image as numpy array.
            track_id_boxes: list of (track_id, cx, cy, w, h) — normalized.
            frame_num: current frame index (written into each Pose).

        Returns:
            Dict mapping track_id → PoseTrack for all active tracks.
        """
        poses = self._estimator.estimate(frame, track_id_boxes)
        for pose in poses:
            pose.frame = frame_num
            track = self._tracks.setdefault(pose.track_id, PoseTrack(track_id=pose.track_id))
            track.poses.append(pose)
            if len(track.poses) > self._max_history:
                track.poses = track.poses[-self._max_history:]
        return self._tracks

    def get_track(self, track_id: int) -> PoseTrack | None:
        return self._tracks.get(track_id)
