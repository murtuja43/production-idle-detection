"""Tests for online ML inference (rolling-window classifier)."""

from __future__ import annotations

import unittest

from src.ml.inference import MlIdleClassifier
from tests.synthetic import make_fake_model, make_zone


class MlIdleClassifierTest(unittest.TestCase):
    """Validate rolling-window warmup and per-zone prediction."""

    def test_not_ready_until_window_is_full(self) -> None:
        model = make_fake_model(["COP"], ["mean", "std"], window_size=4, anomaly=True)
        classifier = MlIdleClassifier(model, [make_zone("COP")])

        for _ in range(3):
            result = classifier.update("COP", 0.5)
            self.assertFalse(result.window_ready)
            self.assertFalse(result.is_anomaly)

        ready = classifier.update("COP", 0.5)
        self.assertTrue(ready.window_ready)
        self.assertTrue(ready.is_anomaly)

    def test_normal_prediction_when_not_anomalous(self) -> None:
        model = make_fake_model(["COP"], ["mean"], window_size=2, anomaly=False, score=0.3)
        classifier = MlIdleClassifier(model, [make_zone("COP")])
        classifier.update("COP", 1.0)
        ready = classifier.update("COP", 1.0)
        self.assertTrue(ready.window_ready)
        self.assertFalse(ready.is_anomaly)
        self.assertAlmostEqual(ready.score, 0.3)

    def test_zone_without_model_abstains(self) -> None:
        model = make_fake_model(["COP"], ["mean"], window_size=2, anomaly=True)
        # Classifier given a zone the model was not trained on.
        classifier = MlIdleClassifier(model, [make_zone("COP"), make_zone("CSK")])
        for _ in range(5):
            result = classifier.update("CSK", 0.5)
            self.assertFalse(result.window_ready)
            self.assertFalse(result.is_anomaly)


if __name__ == "__main__":
    unittest.main()
