"""Smoke test one video before starting the complete evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.models import to_jsonable
from crowd_jaywalking.pipeline import JaywalkingPipeline


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    video_value = os.environ.get("CROWD_JAYWALKING_VIDEO")
    if not video_value:
        raise RuntimeError(
            "Set CROWD_JAYWALKING_VIDEO to a video path before running this smoke test."
        )

    config = ProjectConfig.load(config_path)
    video_path = Path(video_value).resolve()
    smoke_dir = config.path("results") / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    pipeline = JaywalkingPipeline(config)
    result = pipeline.process_video(video_path, smoke_dir / "evidence")
    output_path = smoke_dir / f"{video_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(result), handle, indent=2)

    print(f"Prediction: {result.prediction.value}")
    print(f"Person decisions: {len(result.person_decisions)}")
    print(f"Rejected crossing candidates: {len(result.rejected_candidates)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
