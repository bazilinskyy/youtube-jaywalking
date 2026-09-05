"""Evaluate the frozen crossing classifier on the JAAD test split."""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.crossing_classifier import JAADCrossingClassifierEvaluator


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    config = ProjectConfig.load(config_path)
    JAADCrossingClassifierEvaluator(config).run()


if __name__ == "__main__":
    main()
