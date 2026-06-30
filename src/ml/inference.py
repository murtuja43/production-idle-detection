"""Online ML inference over a streaming motion signal.

Maintains a rolling window of recent motion scores per zone and, once a window is
full, builds a feature vector with the exact same extractor used at training time
and queries the per-zone Isolation Forest.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from src.features.extractor import extract_window_vector
from src.ml.model import IdleAnomalyModel
from src.preprocessing.roi import Zone
from src.utils.config import MlConfig


@dataclass(frozen=True)
class MlZoneResult:
    """ML prediction for a single zone at one frame."""

    window_ready: bool
    is_anomaly: bool
    score: float  # decision function; NaN until the window is full

    @classmethod
    def not_ready(cls) -> "MlZoneResult":
        """Result used before a zone has accumulated a full window."""
        return cls(window_ready=False, is_anomaly=False, score=math.nan)


class MlIdleClassifier:
    """Per-zone rolling-window Isolation Forest inference."""

    def __init__(self, model: IdleAnomalyModel, zones: list[Zone]) -> None:
        self._model = model
        self._feature_names = model.feature_names
        self._window_size = model.window_size
        self._thresholds = {zone.name: zone.motion_threshold for zone in zones}
        self._windows: dict[str, deque[float]] = {
            zone.name: deque(maxlen=self._window_size)
            for zone in zones
            if model.has_zone(zone.name)
        }

    @classmethod
    def from_config(cls, ml_config: MlConfig, zones: list[Zone]) -> "MlIdleClassifier":
        """Load the model referenced by ``ml_config`` and build a classifier."""
        model = IdleAnomalyModel.load(ml_config.model_path, ml_config.metadata_path)
        return cls(model=model, zones=zones)

    @property
    def model(self) -> IdleAnomalyModel:
        """The underlying trained model."""
        return self._model

    def update(self, zone_name: str, motion_score: float) -> MlZoneResult:
        """Push one motion score for a zone and predict when the window is full."""
        window = self._windows.get(zone_name)
        if window is None:
            # No trained estimator for this zone; ML abstains.
            return MlZoneResult.not_ready()

        window.append(motion_score)
        if len(window) < self._window_size:
            return MlZoneResult.not_ready()

        vector = extract_window_vector(
            list(window),
            self._thresholds.get(zone_name, 0.0),
            self._feature_names,
        )
        is_anomaly, score = self._model.predict(zone_name, vector)
        return MlZoneResult(window_ready=True, is_anomaly=is_anomaly, score=score)


def model_exists(ml_config: MlConfig) -> bool:
    """Whether both the model and metadata files referenced by config exist."""
    return Path(ml_config.model_path).exists() and Path(
        ml_config.metadata_path
    ).exists()
