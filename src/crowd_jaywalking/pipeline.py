"""End-to-end per-person jaywalking pipeline."""

from __future__ import annotations

import time
from pathlib import Path

from .config import ProjectConfig
from .crossing import CrossingDetector
from .evidence import EvidenceBuilder
from .models import DecisionLabel, PersonDecision, VideoResult
from .policy import JaywalkingPolicy
from .tracking import PersonTracker
from .vlm import HuggingFaceContextClassifier


class JaywalkingPipeline:
    """Detect crossings first, then classify visible global context per person."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

        # Load the configured local models once and reuse them for every video.
        self.context_classifier = HuggingFaceContextClassifier(config.vlm_settings())
        self.context_classifier.ensure_ready()

        self.tracker = PersonTracker(config.tracking_settings(), config.root)
        self.crossing_detector = CrossingDetector(config.crossing_settings())
        self.evidence_builder = EvidenceBuilder(config.evidence_settings())
        self.policy = JaywalkingPolicy(config.policy_settings())

    def process_video(self, video_path: str | Path, evidence_root: str | Path) -> VideoResult:
        """Process one video and return video and person level decisions."""

        started = time.perf_counter()
        source = Path(video_path).resolve()
        fps, observations = self.tracker.track(source)
        crossings = self.crossing_detector.detect(observations, fps)

        decisions: list[PersonDecision] = []
        for event in crossings.valid_events:
            evidence = self.evidence_builder.build(
                video_path=source,
                event=event,
                observations=observations,
                output_root=evidence_root,
                fps=fps,
            )
            context = self.context_classifier.classify(evidence)
            label, reason = self.policy.decide(context)
            decisions.append(
                PersonDecision(
                    person_id=event.person_id,
                    label=label,
                    reason=reason,
                    event=event,
                    context=context,
                )
            )

        if any(item.label == DecisionLabel.JAYWALKING for item in decisions):
            prediction = DecisionLabel.JAYWALKING
        elif any(item.label == DecisionLabel.UNCERTAIN for item in decisions):
            prediction = DecisionLabel.UNCERTAIN
        else:
            prediction = DecisionLabel.COMPLIANT

        return VideoResult(
            video_path=str(source),
            prediction=prediction,
            person_decisions=decisions,
            rejected_candidates=crossings.rejected_events,
            latency_seconds=round(time.perf_counter() - started, 2),
        )
