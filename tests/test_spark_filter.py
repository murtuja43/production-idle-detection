"""Tests for CMUS spark/glare suppression, including the saturation gate."""

from __future__ import annotations

import unittest

import numpy as np

from src.optical_flow.dense_flow import DenseOpticalFlow
from src.preprocessing.roi import Zone
from src.utils.config import SparkFilterConfig


def _cmus_zone(spark: SparkFilterConfig) -> Zone:
    return Zone(
        name="CMUS", enabled=True, x=0, y=0, width=5, height=5,
        motion_threshold=1.0, idle_duration_seconds=1.0, sensitivity=1.0,
        mask_path=None, spark_filter=spark,
    )


class SparkMaskTest(unittest.TestCase):
    """Validate the binary spark mask under brightness and colour gating."""

    def _gray_with_block(self) -> np.ndarray:
        gray = np.zeros((5, 5), dtype=np.uint8)
        gray[1:3, 1:3] = 255  # bright 2x2 block
        return gray

    def _color_block(self, bgr: tuple[int, int, int]) -> np.ndarray:
        color = np.zeros((5, 5, 3), dtype=np.uint8)
        color[1:3, 1:3] = bgr
        return color

    def test_brightness_only_masks_bright_block(self) -> None:
        spark = SparkFilterConfig(
            enabled=True, brightness_threshold=230,
            min_component_area=1, dilate_iterations=0,
        )
        mask = DenseOpticalFlow._build_spark_mask(
            self._gray_with_block(), None, _cmus_zone(spark)
        )
        self.assertEqual(int(mask.sum()), 4)

    def test_saturation_gate_keeps_coloured_motion(self) -> None:
        spark = SparkFilterConfig(
            enabled=True, brightness_threshold=230, min_component_area=1,
            dilate_iterations=0, saturation_threshold=60,
        )
        zone = _cmus_zone(spark)
        gray = self._gray_with_block()

        # White glare: zero saturation -> still masked as spark.
        white_mask = DenseOpticalFlow._build_spark_mask(
            gray, self._color_block((255, 255, 255)), zone
        )
        self.assertEqual(int(white_mask.sum()), 4)

        # Saturated blue (a coloured moving part): NOT masked.
        blue_mask = DenseOpticalFlow._build_spark_mask(
            gray, self._color_block((255, 0, 0)), zone
        )
        self.assertEqual(int(blue_mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
