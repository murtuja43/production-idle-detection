"""Tests for overlay rendering."""

from __future__ import annotations

import unittest

import numpy as np

from src.utils.config import VisualizationConfig
from src.visualization.overlay import OverlayRenderer, OverlayZone
from tests.synthetic import make_zone


def _config(enabled: bool = True) -> VisualizationConfig:
    return VisualizationConfig(enabled=enabled, font_scale=0.5, line_thickness=2)


class OverlayRendererTest(unittest.TestCase):
    """Validate overlay drawing behavior."""

    def _frame(self) -> np.ndarray:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def _items(self, anomaly: float | None) -> list[OverlayZone]:
        return [
            OverlayZone(
                zone=make_zone("COP"),
                is_idle=True,
                motion_score=0.42,
                threshold=1.0,
                idle_seconds=3.5,
                anomaly_score=anomaly,
            )
        ]

    def test_draw_returns_modified_copy(self) -> None:
        frame = self._frame()
        output = OverlayRenderer(_config()).draw(frame, self._items(None), "combined")
        self.assertEqual(output.shape, frame.shape)
        self.assertFalse(np.array_equal(output, frame))  # something was drawn
        self.assertEqual(int(frame.sum()), 0)  # original untouched

    def test_draw_with_anomaly_score_runs(self) -> None:
        output = OverlayRenderer(_config()).draw(self._frame(), self._items(-0.12), "ml")
        self.assertEqual(output.shape, (240, 320, 3))

    def test_disabled_returns_same_frame(self) -> None:
        frame = self._frame()
        output = OverlayRenderer(_config(enabled=False)).draw(
            frame, self._items(None), "optical_flow"
        )
        self.assertIs(output, frame)


if __name__ == "__main__":
    unittest.main()
