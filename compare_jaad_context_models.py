"""Compare configured Hugging Face VLMs on labelled JAAD context evidence."""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.vlm_comparison import VLMModelComparison


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    VLMModelComparison(ProjectConfig.load(config_path)).run()


if __name__ == "__main__":
    main()
