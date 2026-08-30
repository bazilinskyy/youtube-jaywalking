"""Evaluate the local VLM against completed JAAD context annotations."""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.jaad_context_evaluation import JAADContextBenchmark


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    config = ProjectConfig.load(config_path)
    JAADContextBenchmark(config).run()


if __name__ == "__main__":
    main()
