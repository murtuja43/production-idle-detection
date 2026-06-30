"""Evaluate idle-detection predictions against a ground-truth CSV.

Standalone utility: it reads the detection pipeline's per-frame CSV output and a
ground-truth interval CSV, then reports Accuracy, Precision, Recall, F1, and a
confusion matrix (per zone and overall) and renders a predicted-vs-actual idle
timeline. It does not import or modify the detection pipeline.

Examples:
    python evaluate.py --ground-truth data/ground_truth_example.csv
    python evaluate.py --predictions outputs/idle_detection_log.csv \
        --ground-truth labels.csv --default-label skip --zone CMUS
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.plots import plot_comparison
from src.evaluation.runner import DEFAULT_LABELS, EvaluationResult, evaluate_files
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate idle-detection predictions against ground truth."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/idle_detection_log.csv",
        help="Per-frame detection CSV produced by main.py.",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Ground-truth CSV with labelled timestamp ranges.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for the metrics report and plot.",
    )
    parser.add_argument(
        "--default-label",
        choices=sorted(DEFAULT_LABELS),
        default="active",
        help="Label for frames outside any ground-truth interval "
        "(skip = exclude from scoring).",
    )
    parser.add_argument(
        "--zone",
        default=None,
        help="Evaluate only this zone (default: all zones present).",
    )
    parser.add_argument(
        "--metrics-json", default="evaluation_metrics.json",
        help="Metrics JSON filename (within --output-dir).",
    )
    parser.add_argument(
        "--metrics-csv", default="evaluation_metrics.csv",
        help="Metrics CSV filename (within --output-dir).",
    )
    parser.add_argument(
        "--plot", default="evaluation_comparison.png",
        help="Comparison plot filename (within --output-dir).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot.")
    return parser.parse_args()


def _format_row(name: str, metrics: ClassificationMetrics) -> str:
    return (
        f"{name:<10} {metrics.accuracy:>8.3f} {metrics.precision:>9.3f} "
        f"{metrics.recall:>8.3f} {metrics.f1:>8.3f} "
        f"{metrics.tp:>6} {metrics.fp:>6} {metrics.fn:>6} {metrics.tn:>6}"
    )


def print_report(result: EvaluationResult) -> None:
    """Print a per-zone and overall metrics table to the console."""
    header = (
        f"{'zone':<10} {'accuracy':>8} {'precision':>9} {'recall':>8} "
        f"{'f1':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    )
    print("\nIdle-detection evaluation (positive class = idle)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for zone in result.per_zone:
        print(_format_row(zone.zone, zone.metrics))
    print("-" * len(header))
    print(_format_row("OVERALL", result.overall))
    overall = result.overall
    print(
        f"\nConfusion matrix (overall): "
        f"TP={overall.tp} FP={overall.fp} FN={overall.fn} TN={overall.tn}"
    )
    if result.default_label == "skip":
        print(f"Skipped (unlabelled) frames: {result.skipped_frames}")


def run_evaluation(args: argparse.Namespace) -> EvaluationResult:
    """Run the evaluation workflow and write outputs."""
    result = evaluate_files(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        default_label=args.default_label,
        zone_filter=args.zone,
    )

    output_dir = Path(args.output_dir)
    metrics_json = output_dir / args.metrics_json
    metrics_csv = output_dir / args.metrics_csv
    result.save_json(metrics_json)
    result.save_csv(metrics_csv)

    print_report(result)
    logger.info("Saved metrics to %s and %s", metrics_json, metrics_csv)

    if not args.no_plot:
        plot_path = output_dir / args.plot
        if plot_comparison(result, plot_path):
            logger.info("Saved comparison plot to %s", plot_path)

    return result


def main() -> None:
    """Run the evaluation CLI."""
    configure_logging("INFO")
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
