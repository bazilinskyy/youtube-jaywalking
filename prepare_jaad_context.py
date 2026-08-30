"""Prepare official JAAD crossings for independent context annotation."""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.jaad_context import JAADContextAuditBuilder


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    config = ProjectConfig.load(config_path)
    JAADContextAuditBuilder(config).run()


if __name__ == "__main__":
    main()
