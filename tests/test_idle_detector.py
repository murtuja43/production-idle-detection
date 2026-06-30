"""Tests for stateful idle detection."""

from __future__ import annotations

import unittest

from src.detection.idle_detector import IdleDetector
from src.preprocessing.roi import Zone
from src.utils.config import SparkFilterConfig


def _zone() -> Zone:
    return Zone(
        name="COP",
        enabled=True,
        x=0,
        y=0,
        width=100,
        height=100,
        motion_threshold=1.0,
        idle_duration_seconds=3.0,
        sensitivity=1.0,
        mask_path=None,
        spark_filter=SparkFilterConfig(),
    )


class IdleDetectorTest(unittest.TestCase):
    """Validate idle and active state transitions."""

    def test_zone_becomes_idle_after_duration(self) -> None:
        detector = IdleDetector([_zone()])

        first = detector.update("COP", motion_score=0.2, timestamp_seconds=0.0)
        middle = detector.update("COP", motion_score=0.2, timestamp_seconds=2.0)
        final = detector.update("COP", motion_score=0.2, timestamp_seconds=3.1)

        self.assertFalse(first.is_idle)
        self.assertFalse(middle.is_idle)
        self.assertTrue(final.is_idle)
        self.assertGreaterEqual(final.idle_seconds, 3.0)

    def test_motion_resets_idle_timer(self) -> None:
        detector = IdleDetector([_zone()])

        detector.update("COP", motion_score=0.2, timestamp_seconds=0.0)
        detector.update("COP", motion_score=0.2, timestamp_seconds=3.5)
        active = detector.update("COP", motion_score=2.0, timestamp_seconds=4.0)
        idle_again_start = detector.update(
            "COP", motion_score=0.2, timestamp_seconds=5.0
        )

        self.assertFalse(active.is_idle)
        self.assertTrue(active.is_motion_active)
        self.assertFalse(idle_again_start.is_idle)
        self.assertEqual(idle_again_start.idle_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()

