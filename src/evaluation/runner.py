"""Orchestrate idle-detection evaluation against ground truth.

Reads the detection pipeline's per-frame CSV (its ``is_idle`` column is the final
prediction for whichever mode produced it), aligns each frame with the
ground-truth label for that zone and timestamp, and computes per-zone and overall
metrics. This module is read-only with respect to the detection pipeline.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.ground_truth import GroundTruthInterval, label_at, load_ground_truth
from src.evaluation.metrics import ClassificationMetrics, compute_metrics

# Default-label options for frames not covered by any ground-truth interval.
DEFAULT_LABELS: dict[str, bool | None] = {
    "active": False,
    "idle": True,
    "skip": None,
}


@dataclass(frozen=True)
class ZoneEvaluation:
    """Aligned per-zone series plus its metrics."""

    zone: str
    metrics: ClassificationMetrics
    timestamps: list[float]
    predicted_idle: list[bool]
    actual_idle: list[bool | None]

    def to_dict(self) -> dict[str, object]:
        return {"zone": self.zone, "metrics": self.metrics.to_dict()}


@dataclass(frozen=True)
class EvaluationResult:
    """Overall and per-zone evaluation outcome."""

    overall: ClassificationMetrics
    per_zone: list[ZoneEvaluation]
    default_label: str
    skipped_frames: int
    predictions_path: str
    ground_truth_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "predictions": self.predictions_path,
                "ground_truth": self.ground_truth_path,
                "default_label": self.default_label,
                "skipped_frames": self.skipped_frames,
                "zones_evaluated": [zone.zone for zone in self.per_zone],
            },
            "overall": self.overall.to_dict(),
            "per_zone": [zone.to_dict() for zone in self.per_zone],
        }

    def save_json(self, path: str | Path) -> None:
        """Write the full result (summary + per-zone metrics) as JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)

    def save_csv(self, path: str | Path) -> None:
        """Write a per-zone (plus overall) metrics table as CSV."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "zone", "accuracy", "precision", "recall", "f1",
            "tp", "fp", "fn", "tn", "support",
        ]
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for zone in self.per_zone:
                writer.writerow(_metrics_row(zone.zone, zone.metrics))
            writer.writerow(_metrics_row("OVERALL", self.overall))


def _metrics_row(name: str, metrics: ClassificationMetrics) -> dict[str, object]:
    return {
        "zone": name,
        "accuracy": round(metrics.accuracy, 6),
        "precision": round(metrics.precision, 6),
        "recall": round(metrics.recall, 6),
        "f1": round(metrics.f1, 6),
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "tn": metrics.tn,
        "support": metrics.support,
    }


def load_predictions(
    path: str | Path,
    zone_filter: str | None = None,
) -> dict[str, list[tuple[float, bool]]]:
    """Load per-zone ``(timestamp, predicted_idle)`` rows from a detection CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV does not exist: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"timestamp_seconds", "zone", "is_idle"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Predictions CSV must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}."
            )
        by_zone: dict[str, list[tuple[float, bool]]] = {}
        for row in reader:
            zone = row["zone"]
            if zone_filter is not None and zone != zone_filter:
                continue
            timestamp = float(row["timestamp_seconds"])
            predicted = int(row["is_idle"]) != 0
            by_zone.setdefault(zone, []).append((timestamp, predicted))

    for rows in by_zone.values():
        rows.sort(key=lambda item: item[0])
    return by_zone


def evaluate(
    predictions: dict[str, list[tuple[float, bool]]],
    intervals: Sequence[GroundTruthInterval],
    default_label: str = "active",
    *,
    predictions_path: str = "",
    ground_truth_path: str = "",
) -> EvaluationResult:
    """Align predictions to ground truth and compute metrics."""
    if default_label not in DEFAULT_LABELS:
        raise ValueError(
            f"default_label must be one of {sorted(DEFAULT_LABELS)}; "
            f"got '{default_label}'."
        )
    default = DEFAULT_LABELS[default_label]

    per_zone: list[ZoneEvaluation] = []
    all_true: list[bool] = []
    all_pred: list[bool] = []
    skipped = 0

    for zone in sorted(predictions):
        timestamps: list[float] = []
        predicted_idle: list[bool] = []
        actual_idle: list[bool | None] = []
        scored_true: list[bool] = []
        scored_pred: list[bool] = []

        for timestamp, predicted in predictions[zone]:
            actual = label_at(intervals, zone, timestamp, default)
            timestamps.append(timestamp)
            predicted_idle.append(predicted)
            actual_idle.append(actual)
            if actual is None:
                skipped += 1
                continue
            scored_true.append(actual)
            scored_pred.append(predicted)

        metrics = compute_metrics(scored_true, scored_pred)
        per_zone.append(
            ZoneEvaluation(
                zone=zone,
                metrics=metrics,
                timestamps=timestamps,
                predicted_idle=predicted_idle,
                actual_idle=actual_idle,
            )
        )
        all_true.extend(scored_true)
        all_pred.extend(scored_pred)

    return EvaluationResult(
        overall=compute_metrics(all_true, all_pred),
        per_zone=per_zone,
        default_label=default_label,
        skipped_frames=skipped,
        predictions_path=predictions_path,
        ground_truth_path=ground_truth_path,
    )


def evaluate_files(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    default_label: str = "active",
    zone_filter: str | None = None,
) -> EvaluationResult:
    """Convenience wrapper: load both CSVs from disk and evaluate."""
    predictions = load_predictions(predictions_path, zone_filter=zone_filter)
    if not predictions:
        raise ValueError(
            "No predictions found"
            + (f" for zone '{zone_filter}'." if zone_filter else ".")
        )
    intervals = load_ground_truth(ground_truth_path)
    return evaluate(
        predictions,
        intervals,
        default_label=default_label,
        predictions_path=str(predictions_path),
        ground_truth_path=str(ground_truth_path),
    )
