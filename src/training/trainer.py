"""Isolation Forest training from a feature CSV.

Trains one Isolation Forest per production zone, then persists the bundle and its
metadata via :class:`~src.ml.model.IdleAnomalyModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from sklearn.ensemble import IsolationForest

from src.features.dataset import load_feature_dataset
from src.ml.model import IdleAnomalyModel, ModelMetadata
from src.utils.config import FeatureConfig, TrainingConfig


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of a training run."""

    model: IdleAnomalyModel
    model_path: Path
    metadata_path: Path
    zone_sample_counts: dict[str, int]


def train_from_feature_csv(
    feature_csv: str | Path,
    training_config: TrainingConfig,
    feature_config: FeatureConfig,
    *,
    source: str | None = None,
) -> TrainingResult:
    """Train per-zone Isolation Forests from a feature CSV and persist them.

    The CSV column order is the source of truth for the model's feature layout,
    so inference always feeds features in the same order they were trained on.
    """
    feature_names, matrices = load_feature_dataset(feature_csv)
    if not matrices:
        raise ValueError(f"No training samples found in feature CSV: {feature_csv}")

    models: dict[str, IsolationForest] = {}
    zone_sample_counts: dict[str, int] = {}
    for zone, matrix in matrices.items():
        if matrix.shape[0] == 0:
            continue
        estimator = IsolationForest(
            n_estimators=training_config.n_estimators,
            contamination=training_config.contamination,
            max_samples=training_config.max_samples,
            random_state=training_config.random_state,
        )
        estimator.fit(matrix)
        models[zone] = estimator
        zone_sample_counts[zone] = int(matrix.shape[0])

    if not models:
        raise ValueError("No zone produced any training samples.")

    metadata = ModelMetadata(
        feature_names=list(feature_names),
        window_size=feature_config.window_size,
        step=feature_config.step,
        contamination=training_config.contamination,
        random_state=training_config.random_state,
        n_estimators=training_config.n_estimators,
        max_samples=training_config.max_samples,
        zones=sorted(models),
        zone_sample_counts=zone_sample_counts,
        sklearn_version=sklearn.__version__,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=str(source if source is not None else feature_csv),
    )

    model = IdleAnomalyModel(models=models, metadata=metadata)
    model.save(training_config.model_output, training_config.metadata_output)

    return TrainingResult(
        model=model,
        model_path=Path(training_config.model_output),
        metadata_path=Path(training_config.metadata_output),
        zone_sample_counts=zone_sample_counts,
    )
