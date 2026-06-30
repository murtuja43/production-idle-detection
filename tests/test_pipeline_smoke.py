"""End-to-end smoke test for the Phase 1 video pipeline."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from main import run_pipeline


class PipelineSmokeTest(unittest.TestCase):
    """Run the full pipeline on a small synthetic video."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="idle-detection-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_pipeline_processes_synthetic_video(self) -> None:
        video_path = self.temp_dir / "sample.mp4"
        config_path = self.temp_dir / "config.yaml"
        processed_dir = self.temp_dir / "processed"
        output_dir = self.temp_dir / "outputs"

        self._write_synthetic_video(video_path)
        self._write_test_config(config_path, processed_dir, output_dir)

        run_pipeline(str(config_path), str(video_path))

        self.assertTrue((processed_dir / "sample_processed.mp4").exists())
        csv_path = output_dir / "idle_detection_log.csv"
        self.assertTrue(csv_path.exists())
        self.assertGreater(csv_path.stat().st_size, 100)

    @staticmethod
    def _write_synthetic_video(path: Path) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))
        if not writer.isOpened():
            raise RuntimeError("Could not create synthetic test video.")
        try:
            for frame_index in range(20):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                frame[:] = 30 * (frame_index % 2)
                x = 10 + frame_index * 4
                cv2.rectangle(frame, (x, 50), (x + 30, 90), (255, 255, 255), -1)
                writer.write(frame)
        finally:
            writer.release()

    @staticmethod
    def _write_test_config(
        path: Path,
        processed_dir: Path,
        output_dir: Path,
    ) -> None:
        zone_template = {
            "enabled": True,
            "roi": {"x": 0, "y": 0, "width": 120, "height": 120},
            "motion_threshold": 0.2,
            "idle_duration_seconds": 0.2,
            "sensitivity": 1.0,
        }
        config = {
            "video": {
                "output_dir": str(processed_dir),
                "output_fps": None,
                "codec": "mp4v",
                "resize": {"enabled": False, "width": 320, "height": 240},
            },
            "logging": {
                "level": "WARNING",
                "csv_dir": str(output_dir),
                "csv_filename": "idle_detection_log.csv",
            },
            "optical_flow": {
                "pyr_scale": 0.5,
                "levels": 3,
                "winsize": 15,
                "iterations": 3,
                "poly_n": 5,
                "poly_sigma": 1.2,
                "flags": 0,
            },
            "visualization": {
                "enabled": True,
                "draw_motion_heatmap": False,
                "font_scale": 0.4,
                "line_thickness": 1,
            },
            "zones": {
                "CMUS": {
                    **zone_template,
                    "sensitivity": 0.65,
                    "mask_path": None,
                    "spark_filter": {
                        "enabled": True,
                        "brightness_threshold": 230,
                        "min_component_area": 4,
                        "dilate_iterations": 1,
                    },
                },
                "COP": zone_template,
                "COK": zone_template,
                "CSK": zone_template,
                "CSLT": zone_template,
            },
        }
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file)


if __name__ == "__main__":
    unittest.main()
