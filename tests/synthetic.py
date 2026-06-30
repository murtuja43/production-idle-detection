"""Shared helpers for generating synthetic test videos and zones."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.ml.model import IdleAnomalyModel, ModelMetadata
from src.preprocessing.roi import Zone
from src.utils.config import SparkFilterConfig


def write_synthetic_video(
    path: Path,
    frames: int = 60,
    size: tuple[int, int] = (320, 240),
) -> None:
    """Write a short video with a moving bright rectangle (motion to detect)."""
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic test video.")
    try:
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 10 + (index * 4) % (width - 50)
            cv2.rectangle(frame, (x, 50), (x + 30, 90), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def make_zone(
    name: str = "COP",
    *,
    width: int = 120,
    height: int = 120,
    motion_threshold: float = 1.0,
    idle_duration_seconds: float = 3.0,
    sensitivity: float = 1.0,
) -> Zone:
    """Create a simple test zone at the top-left corner."""
    return Zone(
        name=name,
        enabled=True,
        x=0,
        y=0,
        width=width,
        height=height,
        motion_threshold=motion_threshold,
        idle_duration_seconds=idle_duration_seconds,
        sensitivity=sensitivity,
        mask_path=None,
        spark_filter=SparkFilterConfig(),
    )


class FakeForest:
    """Deterministic stand-in for IsolationForest used in unit tests."""

    def __init__(self, anomaly: bool, score: float = 0.0) -> None:
        self._label = -1 if anomaly else 1
        self._score = score

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self._label, dtype=int)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self._score, dtype=float)


def make_fake_model(
    zone_names: list[str],
    feature_names: list[str],
    window_size: int,
    *,
    anomaly: bool,
    score: float = 0.0,
) -> IdleAnomalyModel:
    """Build an IdleAnomalyModel whose every zone returns a fixed prediction."""
    metadata = ModelMetadata(
        feature_names=list(feature_names),
        window_size=window_size,
        step=1,
        contamination="auto",
        random_state=0,
        n_estimators=1,
        max_samples="auto",
        zones=list(zone_names),
        zone_sample_counts={name: 0 for name in zone_names},
        sklearn_version="test",
        created_at="2026-01-01T00:00:00+00:00",
        source="unit-test",
    )
    models = {name: FakeForest(anomaly=anomaly, score=score) for name in zone_names}
    return IdleAnomalyModel(models=models, metadata=metadata)  # type: ignore[arg-type]
