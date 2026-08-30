"""Tests for the CROWD style configuration file workflow."""

import os
import tempfile
import unittest
from pathlib import Path

from crowd_jaywalking.config import ProjectConfig


class ProjectConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = Path("default.config").read_text(encoding="utf-8")

    def test_uses_default_config_when_active_config_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "default.config").write_text(self.template, encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                config = ProjectConfig.load()
            finally:
                os.chdir(previous)

        self.assertEqual(config.source_path.name, "default.config")
        self.assertEqual(config.get("tracking_model"), "yolo26x.pt")
        self.assertFalse(any(isinstance(value, dict) for value in config.raw.values()))

    def test_active_config_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "default.config").write_text(self.template, encoding="utf-8")
            (root / "config").write_text(self.template, encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                config = ProjectConfig.load()
            finally:
                os.chdir(previous)

        self.assertEqual(config.source_path.name, "config")


if __name__ == "__main__":
    unittest.main()
