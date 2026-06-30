"""Training CLI: generate motion features from video and train Isolation Forests.

Examples:
    # Generate features AND train in one go
    python train.py --config configs/default.yaml --videos data/videos/sample.mp4

    # Only generate the feature CSV
    python train.py --config configs/default.yaml --videos a.mp4 b.mp4 --extract-only

    # Train from an already-generated feature CSV
    python train.py --config configs/default.yaml --skip-extract \
        --features-csv data/processed/features.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from src.features.dataset import FeatureDatasetBuilder
from src.optical_flow.dense_flow import DenseOpticalFlow
from src.pipeline.motion_pipeline import MotionPipeline
from src.preprocessing.roi import ZoneRegistry
from src.training.trainer import train_from_feature_csv
from src.utils.config import AppConfig, FeatureConfig, TrainingConfig, load_config
from src.utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate motion features and train idle-anomaly models."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Input videos used to generate the feature dataset.",
    )
    parser.add_argument("--features-csv", default=None, help="Feature CSV path.")
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--contamination", default=None, help="float or 'auto'.")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Feature names to extract (overrides config).",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only generate the feature CSV; do not train.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Train from an existing feature CSV without regenerating it.",
    )
    return parser.parse_args()


def _resolve_feature_config(config: AppConfig, args: argparse.Namespace) -> FeatureConfig:
    features = config.features
    if args.window_size is not None:
        features = replace(features, window_size=args.window_size)
    if args.step is not None:
        features = replace(features, step=args.step)
    if args.features is not None:
        features = replace(features, features=tuple(args.features))
    return features


def _resolve_training_config(config: AppConfig, args: argparse.Namespace) -> TrainingConfig:
    training = config.training
    if args.features_csv is not None:
        training = replace(training, feature_csv=args.features_csv)
    if args.model_output is not None:
        training = replace(training, model_output=args.model_output)
    if args.metadata_output is not None:
        training = replace(training, metadata_output=args.metadata_output)
    if args.contamination is not None:
        value = args.contamination
        contamination = "auto" if value.lower() == "auto" else float(value)
        training = replace(training, contamination=contamination)
    if args.random_state is not None:
        training = replace(training, random_state=args.random_state)
    if args.n_estimators is not None:
        training = replace(training, n_estimators=args.n_estimators)
    return training


def run_training(args: argparse.Namespace) -> None:
    """Run the feature-generation and/or training workflow."""
    config = load_config(args.config)
    configure_logging(config.logging.level)
    logger = logging.getLogger(__name__)

    if args.extract_only and args.skip_extract:
        raise ValueError("--extract-only and --skip-extract are mutually exclusive.")

    feature_config = _resolve_feature_config(config, args)
    training_config = _resolve_training_config(config, args)
    feature_csv = Path(training_config.feature_csv)

    if not args.skip_extract:
        if not args.videos:
            raise ValueError("--videos is required unless --skip-extract is set.")
        zones = ZoneRegistry.from_config(config.zones)
        flow = DenseOpticalFlow(config.optical_flow)
        pipeline = MotionPipeline(flow, zones.enabled_zones)
        builder = FeatureDatasetBuilder(pipeline, feature_config)

        logger.info("Generating features from %d video(s)", len(args.videos))
        summary = builder.write_csv(args.videos, feature_csv)
        logger.info(
            "Wrote %d feature rows to %s", summary.total_samples, summary.output_path
        )
        for zone, count in sorted(summary.per_zone_counts.items()):
            logger.info("  zone %s: %d windows", zone, count)
        if summary.total_samples == 0:
            raise ValueError(
                "No feature windows were generated. Try a longer video or a "
                "smaller --window-size."
            )

    if args.extract_only:
        logger.info("Extraction complete (--extract-only); skipping training.")
        return

    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    logger.info("Training Isolation Forests from %s", feature_csv)
    result = train_from_feature_csv(
        feature_csv=feature_csv,
        training_config=training_config,
        feature_config=feature_config,
        source=str(feature_csv),
    )
    logger.info("Saved model to %s", result.model_path)
    logger.info("Saved metadata to %s", result.metadata_path)
    for zone, count in sorted(result.zone_sample_counts.items()):
        logger.info("  trained zone %s on %d samples", zone, count)


def main() -> None:
    """Run the training CLI."""
    run_training(parse_args())


if __name__ == "__main__":
    main()
