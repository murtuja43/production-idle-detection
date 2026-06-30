"""Sliding-window motion feature extraction.

Both training (dataset generation) and ML inference build feature vectors from a
window of per-zone motion scores using the exact same functions here, which
guarantees that a trained model is always fed features computed identically to
how it was trained.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# A feature function maps (window scores, zone motion threshold) -> float.
FeatureFn = Callable[[np.ndarray, float], float]


def _mean(scores: np.ndarray, _threshold: float) -> float:
    return float(np.mean(scores))


def _std(scores: np.ndarray, _threshold: float) -> float:
    return float(np.std(scores))


def _max(scores: np.ndarray, _threshold: float) -> float:
    return float(np.max(scores))


def _min(scores: np.ndarray, _threshold: float) -> float:
    return float(np.min(scores))


def _median(scores: np.ndarray, _threshold: float) -> float:
    return float(np.median(scores))


def _range(scores: np.ndarray, _threshold: float) -> float:
    return float(np.max(scores) - np.min(scores))


def _energy(scores: np.ndarray, _threshold: float) -> float:
    """Mean squared magnitude (motion energy within the window)."""
    return float(np.mean(np.square(scores)))


def _active_ratio(scores: np.ndarray, threshold: float) -> float:
    """Fraction of frames whose motion meets the zone's active threshold."""
    return float(np.mean(scores >= threshold))


def _mean_delta(scores: np.ndarray, _threshold: float) -> float:
    """Mean absolute frame-to-frame change (motion jitter)."""
    if scores.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(scores))))


# Canonical registry of available window features.
AVAILABLE_FEATURES: dict[str, FeatureFn] = {
    "mean": _mean,
    "std": _std,
    "max": _max,
    "min": _min,
    "median": _median,
    "range": _range,
    "energy": _energy,
    "active_ratio": _active_ratio,
    "mean_delta": _mean_delta,
}

# Default feature set used when none is configured.
DEFAULT_FEATURES: tuple[str, ...] = (
    "mean",
    "std",
    "max",
    "min",
    "active_ratio",
    "mean_delta",
)


def validate_feature_names(feature_names: Sequence[str]) -> None:
    """Raise ValueError if any requested feature is unknown."""
    unknown = [name for name in feature_names if name not in AVAILABLE_FEATURES]
    if unknown:
        raise ValueError(
            f"Unknown feature(s): {unknown}. "
            f"Available: {sorted(AVAILABLE_FEATURES)}."
        )


def extract_window_features(
    scores: Sequence[float],
    threshold: float,
    feature_names: Sequence[str],
) -> dict[str, float]:
    """Compute the selected features for one window of motion scores.

    Args:
        scores: Per-frame motion magnitudes within the window.
        threshold: The zone's motion threshold (used by threshold-aware features
            such as ``active_ratio``).
        feature_names: Ordered names of features to compute.

    Returns:
        Mapping of feature name to value, in the requested order.
    """
    validate_feature_names(feature_names)
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot extract features from an empty window.")
    return {name: AVAILABLE_FEATURES[name](array, threshold) for name in feature_names}


def extract_window_vector(
    scores: Sequence[float],
    threshold: float,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Return the selected features as an ordered 1-D float vector."""
    features = extract_window_features(scores, threshold, feature_names)
    return np.array([features[name] for name in feature_names], dtype=np.float64)
