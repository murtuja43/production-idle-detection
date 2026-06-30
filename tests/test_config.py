"""Tests for YAML configuration loading."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.utils.config import REQUIRED_ZONES, load_config


class ConfigLoadingTest(unittest.TestCase):
    """Validate the default project configuration."""

    def test_default_config_loads_required_zones(self) -> None:
        config = load_config(Path("configs/default.yaml"))

        self.assertEqual(set(config.zones.keys()), set(REQUIRED_ZONES))
        self.assertEqual(config.zones["CMUS"].name, "CMUS")
        self.assertTrue(config.zones["CMUS"].spark_filter.enabled)
        self.assertLess(config.zones["CMUS"].sensitivity, 1.0)

    def test_default_output_paths_are_present(self) -> None:
        config = load_config(Path("configs/default.yaml"))

        self.assertEqual(config.video.output_dir, "data/processed")
        self.assertEqual(config.logging.csv_dir, "outputs")
        self.assertTrue(config.logging.csv_filename.endswith(".csv"))


if __name__ == "__main__":
    unittest.main()

