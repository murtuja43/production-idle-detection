"""End-to-end smoke test: train a model, then run ML and combined inference."""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

import main
import train
from tests.synthetic import write_synthetic_video


class MlPipelineSmokeTest(unittest.TestCase):
    """Train on a synthetic video and run inference in ml/combined modes."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ml-smoke-test-"))
        self.video = self.temp_dir / "sample.mp4"
        write_synthetic_video(self.video, frames=80)
        self.processed_dir = self.temp_dir / "processed"
        self.outputs_dir = self.temp_dir / "outputs"
        self.model_path = self.temp_dir / "model.joblib"
        self.metadata_path = self.temp_dir / "model.metadata.json"
        self.feature_csv = self.temp_dir / "features.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _write_config(self, mode: str) -> Path:
        zone_template = {
            "enabled": True,
            "roi": {"x": 0, "y": 0, "width": 120, "height": 120},
            "motion_threshold": 0.2,
            "idle_duration_seconds": 0.2,
            "sensitivity": 1.0,
        }
        config = {
            "video": {
                "output_dir": str(self.processed_dir),
                "output_fps": None,
                "codec": "mp4v",
                "resize": {"enabled": False, "width": 320, "height": 240},
            },
            "logging": {
                "level": "WARNING",
                "csv_dir": str(self.outputs_dir),
                "csv_filename": "idle_detection_log.csv",
            },
            "optical_flow": {
                "pyr_scale": 0.5, "levels": 3, "winsize": 15,
                "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0,
            },
            "visualization": {
                "enabled": True, "draw_motion_heatmap": False,
                "font_scale": 0.4, "line_thickness": 1,
            },
            "detection": {"mode": mode},
            "features": {
                "window_size": 10,
                "step": 5,
                "features": ["mean", "std", "max", "active_ratio"],
            },
            "ml": {
                "model_path": str(self.model_path),
                "metadata_path": str(self.metadata_path),
                "combine": {"strategy": "and"},
            },
            "training": {
                "contamination": "auto",
                "random_state": 42,
                "n_estimators": 40,
                "max_samples": "auto",
                "feature_csv": str(self.feature_csv),
                "model_output": str(self.model_path),
                "metadata_output": str(self.metadata_path),
            },
            "zones": {
                "CMUS": {
                    **zone_template,
                    "sensitivity": 0.65,
                    "mask_path": None,
                    "spark_filter": {"enabled": True, "brightness_threshold": 230,
                                     "min_component_area": 4, "dilate_iterations": 1},
                },
                "COP": zone_template,
                "COK": zone_template,
                "CSK": zone_template,
                "CSLT": zone_template,
            },
        }
        path = self.temp_dir / "config.yaml"
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file)
        return path

    def _train(self, config_path: Path) -> None:
        args = argparse.Namespace(
            config=str(config_path), videos=[str(self.video)], features_csv=None,
            model_output=None, metadata_output=None, contamination=None,
            random_state=None, n_estimators=None, window_size=None, step=None,
            features=None, extract_only=False, skip_extract=False,
        )
        train.run_training(args)

    def test_train_and_run_ml_and_combined(self) -> None:
        config_path = self._write_config("combined")
        self._train(config_path)

        self.assertTrue(self.feature_csv.exists())
        self.assertTrue(self.model_path.exists())
        self.assertTrue(self.metadata_path.exists())

        # Combined mode (from config).
        main.run_pipeline(config_path=str(config_path), video_path=str(self.video))
        processed = self.processed_dir / "sample_processed.mp4"
        csv_path = self.outputs_dir / "idle_detection_log.csv"
        self.assertTrue(processed.exists())
        self.assertTrue(csv_path.exists())

        with csv_path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertIn("ml_is_anomaly", rows[0])
        self.assertIn("mode", rows[0])
        self.assertTrue(any(row["mode"] == "combined" for row in rows))
        self.assertTrue(any(row["ml_window_ready"] == "1" for row in rows))

        # ML mode via CLI override.
        main.run_pipeline(
            config_path=str(config_path), video_path=str(self.video), mode="ml"
        )
        with csv_path.open(encoding="utf-8") as file:
            ml_rows = list(csv.DictReader(file))
        self.assertTrue(any(row["mode"] == "ml" for row in ml_rows))

    def test_ml_mode_without_model_raises(self) -> None:
        config_path = self._write_config("ml")
        with self.assertRaises(FileNotFoundError):
            main.run_pipeline(config_path=str(config_path), video_path=str(self.video))


if __name__ == "__main__":
    unittest.main()
