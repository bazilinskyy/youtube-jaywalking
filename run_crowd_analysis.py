"""Download mapped CROWD clips and run the frozen jaywalking method."""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.crowd_analysis import CrowdAnalysisRunner


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    CrowdAnalysisRunner(ProjectConfig.load(config_path)).run()


if __name__ == "__main__":
    main()
