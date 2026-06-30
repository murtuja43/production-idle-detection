"""Feature dataset generation from video.

Drives the shared :class:`MotionPipeline` over one or more videos, aggregates the
per-zone motion scores into sliding windows, and writes a feature CSV that the
trainer consumes. No optical-flow logic is duplicated here.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.extractor import extract_window_features, validate_feature_names
from src.pipeline.motion_pipeline import MotionPipeline
from src.preprocessing.video_loader import VideoProcessor
from src.utils.config import FeatureConfig

# Non-feature columns written alongside every window row.
META_COLUMNS: tuple[str, ...] = (
    "source",
    "zone",
    "window_index",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
)


@dataclass(frozen=True)
class WindowSample:
    """One sliding-window feature sample for a single zone."""

    zone: str
    source: str
    window_index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    features: dict[str, float]


@dataclass(frozen=True)
class DatasetSummary:
    """Summary of a generated feature dataset."""

    total_samples: int
    per_zone_counts: dict[str, int]
    feature_names: tuple[str, ...]
    output_path: Path


class FeatureDatasetBuilder:
    """Build sliding-window feature datasets from videos via MotionPipeline."""

    def __init__(self, pipeline: MotionPipeline, feature_config: FeatureConfig) -> None:
        self._pipeline = pipeline
        self._window_size = feature_config.window_size
        self._step = feature_config.step
        self._feature_names = tuple(feature_config.features)
        validate_feature_names(self._feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered feature names produced by this builder."""
        return self._feature_names

    def iter_windows(self, video_path: str | Path) -> Iterator[WindowSample]:
        """Yield window samples for every zone of a single video."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video does not exist: {path}")

        zones = self._pipeline.zones
        thresholds = {zone.name: zone.motion_threshold for zone in zones}
        scores: dict[str, list[float]] = {zone.name: [] for zone in zones}
        frames: dict[str, list[int]] = {zone.name: [] for zone in zones}
        times: dict[str, list[float]] = {zone.name: [] for zone in zones}

        with VideoProcessor(path) as processor:
            for frame_motion in self._pipeline.iter_motion(processor):
                for zone_motion in frame_motion.zone_motions:
                    name = zone_motion.zone.name
                    scores[name].append(zone_motion.motion_score)
                    frames[name].append(frame_motion.frame_index)
                    times[name].append(frame_motion.timestamp_seconds)

        for zone in zones:
            name = zone.name
            zone_scores = scores[name]
            count = len(zone_scores)
            window_index = 0
            for start in range(0, count - self._window_size + 1, self._step):
                end = start + self._window_size
                features = extract_window_features(
                    zone_scores[start:end],
                    thresholds[name],
                    self._feature_names,
                )
                yield WindowSample(
                    zone=name,
                    source=path.name,
                    window_index=window_index,
                    start_frame=frames[name][start],
                    end_frame=frames[name][end - 1],
                    start_time=times[name][start],
                    end_time=times[name][end - 1],
                    features=features,
                )
                window_index += 1

    def write_csv(
        self,
        video_paths: Iterable[str | Path],
        output_path: str | Path,
    ) -> DatasetSummary:
        """Generate features for all videos and write them to ``output_path``."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(META_COLUMNS) + list(self._feature_names)
        per_zone_counts: dict[str, int] = {}
        total = 0

        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for video_path in video_paths:
                for sample in self.iter_windows(video_path):
                    row: dict[str, object] = {
                        "source": sample.source,
                        "zone": sample.zone,
                        "window_index": sample.window_index,
                        "start_frame": sample.start_frame,
                        "end_frame": sample.end_frame,
                        "start_time": f"{sample.start_time:.3f}",
                        "end_time": f"{sample.end_time:.3f}",
                    }
                    for name in self._feature_names:
                        row[name] = f"{sample.features[name]:.6f}"
                    writer.writerow(row)
                    per_zone_counts[sample.zone] = (
                        per_zone_counts.get(sample.zone, 0) + 1
                    )
                    total += 1

        return DatasetSummary(
            total_samples=total,
            per_zone_counts=per_zone_counts,
            feature_names=self._feature_names,
            output_path=output,
        )


def load_feature_dataset(
    csv_path: str | Path,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Read a feature CSV into ordered feature names and per-zone matrices.

    Returns:
        ``(feature_names, {zone: matrix})`` where each matrix has shape
        ``(n_samples, n_features)`` with columns in ``feature_names`` order.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Feature CSV is empty: {path}")
        feature_names = [
            name for name in reader.fieldnames if name not in META_COLUMNS
        ]
        if not feature_names:
            raise ValueError(f"Feature CSV has no feature columns: {path}")

        rows_by_zone: dict[str, list[list[float]]] = {}
        for row in reader:
            zone = row["zone"]
            rows_by_zone.setdefault(zone, []).append(
                [float(row[name]) for name in feature_names]
            )

    matrices = {
        zone: np.asarray(rows, dtype=np.float64)
        for zone, rows in rows_by_zone.items()
    }
    return feature_names, matrices


def feature_names_from_config(features: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize configured feature names."""
    validate_feature_names(features)
    return tuple(features)
