"""Run the complete JAAD evaluation using the project config."""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.evaluation import EvaluationRunner


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    config = ProjectConfig.load(config_path)
    EvaluationRunner(config).run()


if __name__ == "__main__":
    main()
