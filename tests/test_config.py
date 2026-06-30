"""Tests for YAML configuration loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

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


class Phase2ConfigTest(unittest.TestCase):
    """Validate Phase 2 detection/features/ml/training config sections."""

    def test_default_config_has_phase2_defaults(self) -> None:
        config = load_config(Path("configs/default.yaml"))

        self.assertEqual(config.detection.mode, "optical_flow")
        self.assertGreaterEqual(config.features.window_size, 2)
        self.assertIn("mean", config.features.features)
        self.assertEqual(config.ml.combine.strategy, "and")
        self.assertEqual(config.training.contamination, "auto")

    def _load_with_overrides(self, **sections: object) -> None:
        base = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
        for key, value in sections.items():
            base[key] = value
        path = Path(tempfile.mkdtemp()) / "config.yaml"
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(base, file)
        load_config(path)

    def test_legacy_config_without_phase2_sections_loads(self) -> None:
        base = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
        for key in ("detection", "features", "ml", "training"):
            base.pop(key, None)
        path = Path(tempfile.mkdtemp()) / "legacy.yaml"
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(base, file)

        config = load_config(path)
        self.assertEqual(config.detection.mode, "optical_flow")

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._load_with_overrides(detection={"mode": "bogus"})

    def test_invalid_combine_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._load_with_overrides(ml={"combine": {"strategy": "xor"}})

    def test_unknown_feature_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._load_with_overrides(
                features={"window_size": 30, "step": 15, "features": ["bogus"]}
            )

    def test_out_of_range_contamination_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._load_with_overrides(training={"contamination": 0.9})

    def test_invalid_max_samples_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._load_with_overrides(training={"max_samples": 2.5})

    def test_invalid_spark_brightness_raises(self) -> None:
        base = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
        base["zones"]["CMUS"]["spark_filter"]["brightness_threshold"] = 999
        path = Path(tempfile.mkdtemp()) / "config.yaml"
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(base, file)
        with self.assertRaises(ValueError):
            load_config(path)


if __name__ == "__main__":
    unittest.main()

