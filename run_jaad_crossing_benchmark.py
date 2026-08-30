"""Validate tracking and crossing detection against official JAAD labels."""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.jaad_benchmark import JAADCrossingBenchmark


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    config = ProjectConfig.load(config_path)
    JAADCrossingBenchmark(config).run()


if __name__ == "__main__":
    main()
