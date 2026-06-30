"""Trained idle-anomaly model bundle and its metadata.

A model is one :class:`~sklearn.ensemble.IsolationForest` per production zone,
because zones have very different motion characteristics (e.g. CMUS welding vs a
conveyor). The per-zone estimators are bundled into a single joblib file and the
training metadata is persisted alongside as JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class ModelMetadata:
    """Provenance and feature definition for a trained model."""

    feature_names: list[str]
    window_size: int
    step: int
    contamination: float | str
    random_state: int
    n_estimators: int
    max_samples: int | float | str
    zones: list[str]
    zone_sample_counts: dict[str, int]
    sklearn_version: str
    created_at: str
    source: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ModelMetadata":
        """Build metadata from a parsed JSON dict."""
        return cls(
            feature_names=list(data["feature_names"]),  # type: ignore[arg-type]
            window_size=int(data["window_size"]),  # type: ignore[arg-type]
            step=int(data["step"]),  # type: ignore[arg-type]
            contamination=data["contamination"],  # type: ignore[assignment]
            random_state=int(data["random_state"]),  # type: ignore[arg-type]
            n_estimators=int(data["n_estimators"]),  # type: ignore[arg-type]
            max_samples=data["max_samples"],  # type: ignore[assignment]
            zones=list(data["zones"]),  # type: ignore[arg-type]
            zone_sample_counts=dict(data["zone_sample_counts"]),  # type: ignore[arg-type]
            sklearn_version=str(data["sklearn_version"]),
            created_at=str(data["created_at"]),
            source=str(data["source"]),
        )

    def save_json(self, path: str | Path) -> None:
        """Write metadata to ``path`` as pretty-printed JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelMetadata":
        """Read metadata from a JSON file."""
        meta_path = Path(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


class IdleAnomalyModel:
    """A bundle of per-zone Isolation Forests plus its training metadata."""

    def __init__(
        self,
        models: dict[str, IsolationForest],
        metadata: ModelMetadata,
    ) -> None:
        self._models = models
        self.metadata = metadata

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered feature names the model expects."""
        return tuple(self.metadata.feature_names)

    @property
    def window_size(self) -> int:
        """Window size (frames) the model was trained with."""
        return self.metadata.window_size

    def zones(self) -> list[str]:
        """Zones that have a trained estimator."""
        return sorted(self._models)

    def has_zone(self, zone: str) -> bool:
        """Whether a trained estimator exists for ``zone``."""
        return zone in self._models

    def predict(self, zone: str, vector: np.ndarray) -> tuple[bool, float]:
        """Predict for one zone's feature vector.

        Returns:
            ``(is_anomaly, score)`` where ``score`` is the Isolation Forest
            decision function (higher = more normal, lower = more anomalous).
        """
        if zone not in self._models:
            raise KeyError(f"No trained model for zone: {zone}")
        features = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        expected = len(self.metadata.feature_names)
        if features.shape[1] != expected:
            raise ValueError(
                f"Zone '{zone}' expected {expected} features, "
                f"got {features.shape[1]}."
            )
        label = int(self._models[zone].predict(features)[0])
        score = float(self._models[zone].decision_function(features)[0])
        return label == -1, score

    def save(self, model_path: str | Path, metadata_path: str | Path) -> None:
        """Persist the model bundle (joblib) and metadata (JSON)."""
        model_file = Path(model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"models": self._models, "metadata": self.metadata.to_dict()},
            model_file,
        )
        self.metadata.save_json(metadata_path)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        metadata_path: str | Path,
    ) -> "IdleAnomalyModel":
        """Load a model bundle and its metadata from disk."""
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        bundle = joblib.load(model_file)
        metadata = ModelMetadata.load_json(metadata_path)
        return cls(models=bundle["models"], metadata=metadata)
