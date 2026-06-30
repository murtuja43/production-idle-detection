"""Visualization comparing predicted vs actual idle periods."""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from pathlib import Path

from src.evaluation.runner import EvaluationResult, ZoneEvaluation

logger = logging.getLogger(__name__)

_PRED_COLOR = "#1565c0"
_ACTUAL_COLOR = "#c62828"


def _median_period(timestamps: Sequence[float]) -> float:
    if len(timestamps) < 2:
        return 1.0
    diffs = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    return statistics.median(diffs) if diffs else 1.0


def _to_intervals(
    timestamps: Sequence[float],
    flags: Sequence[bool | None],
    period: float,
) -> list[tuple[float, float]]:
    """Convert a boolean idle series into (start, width) spans for broken_barh."""
    spans: list[tuple[float, float]] = []
    start: float | None = None
    last = 0.0
    for timestamp, flag in zip(timestamps, flags):
        if flag:  # None and False both end a span
            if start is None:
                start = timestamp
            last = timestamp
        elif start is not None:
            spans.append((start, max(period, last + period - start)))
            start = None
    if start is not None:
        spans.append((start, max(period, last + period - start)))
    return spans


def plot_comparison(
    result: EvaluationResult,
    output_path: str | Path,
    title: str | None = None,
) -> bool:
    """Render a predicted-vs-actual idle timeline per zone. Returns success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as error:  # pragma: no cover - environment dependent
        logger.warning("Skipping evaluation plot (matplotlib unavailable): %s", error)
        return False

    zones = result.per_zone
    if not zones:
        logger.warning("No zones to plot.")
        return False

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(zones), 1,
        figsize=(11, 1.7 * len(zones) + 1.2),
        sharex=True,
        squeeze=False,
    )
    for axis, zone in zip(axes[:, 0], zones):
        _plot_zone(axis, zone)

    axes[-1, 0].set_xlabel("Time (s)")
    figure.legend(
        handles=[
            Patch(color=_ACTUAL_COLOR, label="Actual idle"),
            Patch(color=_PRED_COLOR, label="Predicted idle"),
        ],
        loc="upper right",
        ncol=2,
    )
    figure.suptitle(title or "Predicted vs actual idle periods", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=120)
    plt.close(figure)
    return True


def _plot_zone(axis, zone: ZoneEvaluation) -> None:
    period = _median_period(zone.timestamps)
    actual_spans = _to_intervals(zone.timestamps, zone.actual_idle, period)
    pred_spans = _to_intervals(zone.timestamps, zone.predicted_idle, period)

    axis.broken_barh(actual_spans, (1.1, 0.8), facecolors=_ACTUAL_COLOR)
    axis.broken_barh(pred_spans, (0.1, 0.8), facecolors=_PRED_COLOR)
    axis.set_yticks([0.5, 1.5])
    axis.set_yticklabels(["Predicted", "Actual"])
    axis.set_ylim(0, 2)
    if zone.timestamps:
        axis.set_xlim(zone.timestamps[0], zone.timestamps[-1] + period)
    metrics = zone.metrics
    axis.set_title(
        f"{zone.zone}   "
        f"F1={metrics.f1:.2f}  P={metrics.precision:.2f}  "
        f"R={metrics.recall:.2f}  Acc={metrics.accuracy:.2f}",
        fontsize=9,
        loc="left",
    )
    axis.grid(True, axis="x", linestyle=":", alpha=0.4)
