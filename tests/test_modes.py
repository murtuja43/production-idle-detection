"""Tests for combine logic and the mode evaluator across all modes."""

from __future__ import annotations

import unittest

import numpy as np

from src.detection.combined import combine_idle
from src.detection.evaluator import ModeEvaluator
from src.detection.idle_detector import IdleDetector
from src.pipeline.motion_pipeline import FrameMotion, ZoneMotion
from src.preprocessing.roi import Zone
from tests.synthetic import make_fake_model, make_zone

_FRAME = np.zeros((1, 1, 3), dtype=np.uint8)
_GRAY = np.zeros((1, 1), dtype=np.uint8)


def _frame_motion(zone: Zone, score: float, index: int) -> FrameMotion:
    return FrameMotion(
        frame_index=index,
        timestamp_seconds=float(index),
        frame=_FRAME,
        gray=_GRAY,
        zone_motions=[ZoneMotion(zone=zone, motion_score=score)],
    )


def _feed(evaluator: ModeEvaluator, zone: Zone, scores: list[float]) -> list[bool]:
    """Feed a sequence of motion scores; return final is_idle per frame."""
    decisions = []
    for index, score in enumerate(scores):
        evaluations = evaluator.evaluate_frame(_frame_motion(zone, score, index))
        decisions.append(evaluations[0].is_idle)
    return decisions


class CombineIdleTest(unittest.TestCase):
    """Validate the boolean fusion of optical-flow and ML signals."""

    def test_and_strategy(self) -> None:
        self.assertTrue(combine_idle(True, True, "and"))
        self.assertFalse(combine_idle(True, False, "and"))
        self.assertFalse(combine_idle(False, True, "and"))
        self.assertFalse(combine_idle(False, False, "and"))

    def test_or_strategy(self) -> None:
        self.assertTrue(combine_idle(True, False, "or"))
        self.assertTrue(combine_idle(False, True, "or"))
        self.assertFalse(combine_idle(False, False, "or"))

    def test_invalid_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            combine_idle(True, True, "xor")


class ModeEvaluatorTest(unittest.TestCase):
    """Validate per-mode final decisions and backward compatibility."""

    def _zone(self) -> Zone:
        # idle_duration 0 => idle the moment motion drops below threshold.
        return make_zone("COP", motion_threshold=1.0, idle_duration_seconds=0.0)

    def test_ml_mode_requires_classifier(self) -> None:
        with self.assertRaises(ValueError):
            ModeEvaluator("ml", IdleDetector([self._zone()]), classifier=None)

    def test_optical_flow_mode_matches_raw_detector(self) -> None:
        zone = self._zone()
        evaluator = ModeEvaluator("optical_flow", IdleDetector([zone]))
        reference = IdleDetector([zone])

        scores = [2.0, 0.0, 0.0, 2.0, 0.0]
        final = _feed(evaluator, zone, scores)
        expected = [
            reference.update("COP", score, float(i)).is_idle
            for i, score in enumerate(scores)
        ]
        self.assertEqual(final, expected)

    def test_ml_mode_uses_anomaly_when_ready(self) -> None:
        from src.ml.inference import MlIdleClassifier

        zone = self._zone()
        model = make_fake_model(["COP"], ["mean"], window_size=1, anomaly=True)
        evaluator = ModeEvaluator(
            "ml", IdleDetector([zone]), MlIdleClassifier(model, [zone])
        )
        # High motion => optical flow NOT idle, but ML anomaly => final idle.
        final = _feed(evaluator, zone, [5.0, 5.0])
        self.assertEqual(final, [True, True])

    def test_ml_mode_falls_back_to_optical_flow_until_ready(self) -> None:
        from src.ml.inference import MlIdleClassifier

        zone = self._zone()
        model = make_fake_model(["COP"], ["mean"], window_size=3, anomaly=False)
        evaluator = ModeEvaluator(
            "ml", IdleDetector([zone]), MlIdleClassifier(model, [zone])
        )
        # Low motion => optical flow idle. ML (not anomalous) only kicks in once
        # the 3-frame window is full, flipping the decision to active.
        final = _feed(evaluator, zone, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(final, [True, True, False, False])

    def test_combined_and_requires_both(self) -> None:
        from src.ml.inference import MlIdleClassifier

        zone = self._zone()
        anomaly_model = make_fake_model(["COP"], ["mean"], window_size=1, anomaly=True)
        evaluator = ModeEvaluator(
            "combined",
            IdleDetector([zone]),
            MlIdleClassifier(anomaly_model, [zone]),
            combine_strategy="and",
        )
        # Low motion (OF idle) + ML anomaly => idle.
        self.assertEqual(_feed(evaluator, zone, [0.0]), [True])

        evaluator_active = ModeEvaluator(
            "combined",
            IdleDetector([self._zone()]),
            MlIdleClassifier(anomaly_model, [self._zone()]),
            combine_strategy="and",
        )
        # High motion (OF active) + ML anomaly => not idle under 'and'.
        self.assertEqual(_feed(evaluator_active, self._zone(), [5.0]), [False])

    def test_combined_or_fires_on_either(self) -> None:
        from src.ml.inference import MlIdleClassifier

        zone = self._zone()
        anomaly_model = make_fake_model(["COP"], ["mean"], window_size=1, anomaly=True)
        evaluator = ModeEvaluator(
            "combined",
            IdleDetector([zone]),
            MlIdleClassifier(anomaly_model, [zone]),
            combine_strategy="or",
        )
        # High motion (OF active) but ML anomaly => idle under 'or'.
        self.assertEqual(_feed(evaluator, zone, [5.0]), [True])


if __name__ == "__main__":
    unittest.main()
