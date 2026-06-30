"""Command-line entry point for idle detection (optical-flow / ml / combined)."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from src.detection.evaluator import ModeEvaluator
from src.detection.idle_detector import IdleDetector
from src.ml.inference import MlIdleClassifier, model_exists
from src.optical_flow.dense_flow import DenseOpticalFlow
from src.pipeline.motion_pipeline import MotionPipeline
from src.preprocessing.roi import ZoneRegistry
from src.preprocessing.video_loader import VideoProcessor
from src.reporting.report import ReportAggregator
from src.utils.config import AppConfig, MlConfig, load_config
from src.utils.csv_logger import CsvIdleLogger
from src.utils.logger import configure_logging
from src.visualization.overlay import OverlayRenderer, OverlayZone


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Detect industrial production idle time from video."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--mode",
        choices=["optical_flow", "ml", "combined"],
        default=None,
        help="Detection mode (overrides detection.mode in the config).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the trained model (overrides ml.model_path).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to the model metadata (overrides ml.metadata_path).",
    )
    return parser.parse_args()


def _resolve_ml_config(config: AppConfig, model: str | None, metadata: str | None) -> MlConfig:
    """Apply CLI overrides for model/metadata paths."""
    ml = config.ml
    if model is not None:
        ml = replace(ml, model_path=model)
    if metadata is not None:
        ml = replace(ml, metadata_path=metadata)
    return ml


def run_pipeline(
    config_path: str,
    video_path: str,
    mode: str | None = None,
    model: str | None = None,
    metadata: str | None = None,
) -> None:
    """Run the idle-detection pipeline in the selected mode."""
    config = load_config(config_path)
    configure_logging(config.logging.level)
    logger = logging.getLogger(__name__)

    active_mode = mode or config.detection.mode
    ml_config = _resolve_ml_config(config, model, metadata)

    input_video = Path(video_path)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    zones = ZoneRegistry.from_config(config.zones)
    flow = DenseOpticalFlow(config.optical_flow)
    detector = IdleDetector.from_zones(zones)
    overlay = OverlayRenderer(config.visualization)
    pipeline = MotionPipeline(flow, zones.enabled_zones)

    classifier: MlIdleClassifier | None = None
    if active_mode in ("ml", "combined"):
        if not model_exists(ml_config):
            raise FileNotFoundError(
                f"Mode '{active_mode}' needs a trained model at "
                f"'{ml_config.model_path}' and metadata at "
                f"'{ml_config.metadata_path}'. Train one first with train.py."
            )
        classifier = MlIdleClassifier.from_config(ml_config, zones.enabled_zones)

    evaluator = ModeEvaluator(
        mode=active_mode,
        detector=detector,
        classifier=classifier,
        combine_strategy=config.ml.combine.strategy,
    )

    output_dir = Path(config.video.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{input_video.stem}_processed.mp4"

    csv_dir = Path(config.logging.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / config.logging.csv_filename

    processor = VideoProcessor(
        input_path=input_video,
        output_path=output_video,
        codec=config.video.codec,
        output_fps=config.video.output_fps,
        resize_enabled=config.video.resize.enabled,
        resize_width=config.video.resize.width,
        resize_height=config.video.resize.height,
    )

    logger.info("Starting idle detection (mode=%s) for %s", active_mode, input_video)
    logger.info("Writing processed video to %s", output_video)
    logger.info("Writing CSV log to %s", csv_path)

    include_ml = active_mode != "optical_flow"
    with processor, CsvIdleLogger(csv_path, include_ml=include_ml) as csv_logger:
        # fps is known only after the processor is opened (inside the context).
        aggregator = ReportAggregator(
            zone_names=[zone.name for zone in zones.enabled_zones],
            mode=active_mode,
            fps=processor.fps,
        )
        progress = tqdm(
            total=processor.frame_count or None,
            desc="Processing frames",
            unit="frame",
        )
        try:
            for frame_motion in pipeline.iter_motion(processor):
                overlay_items = []
                for evaluation in evaluator.evaluate_frame(frame_motion):
                    ml_result = evaluation.ml_result
                    anomaly_score = (
                        ml_result.score
                        if ml_result is not None and ml_result.window_ready
                        else None
                    )
                    is_anomaly = bool(
                        ml_result is not None
                        and ml_result.window_ready
                        and ml_result.is_anomaly
                    )
                    overlay_items.append(
                        OverlayZone(
                            zone=evaluation.zone,
                            is_idle=evaluation.is_idle,
                            motion_score=evaluation.motion_score,
                            threshold=evaluation.zone.motion_threshold,
                            idle_seconds=evaluation.final_state.idle_seconds,
                            anomaly_score=anomaly_score,
                        )
                    )
                    aggregator.update(
                        zone_name=evaluation.zone.name,
                        motion_score=evaluation.motion_score,
                        is_idle=evaluation.is_idle,
                        is_anomaly=is_anomaly,
                    )
                    csv_logger.write(
                        frame_index=frame_motion.frame_index,
                        timestamp_seconds=frame_motion.timestamp_seconds,
                        zone=evaluation.zone,
                        motion_score=evaluation.motion_score,
                        state=evaluation.final_state,
                        mode=evaluation.mode,
                        optical_flow_idle=evaluation.optical_flow_state.is_idle,
                        ml_result=ml_result,
                    )

                rendered = overlay.draw(frame_motion.frame, overlay_items, active_mode)
                processor.write(rendered)
                progress.update(1)
        finally:
            progress.close()

    report = aggregator.build(video=input_video.name)
    report_csv = csv_dir / config.logging.report_csv_filename
    report_json = csv_dir / config.logging.report_json_filename
    report.save_csv(report_csv)
    report.save_json(report_json)
    if config.logging.report_chart:
        report.save_chart(csv_dir / config.logging.report_chart_filename)

    logger.info("Finished processing %s", input_video)
    logger.info("Processed video saved to %s", output_video)
    logger.info("CSV log saved to %s", csv_path)
    logger.info("Report saved to %s and %s", report_csv, report_json)


def main() -> None:
    """Run the CLI."""
    args = parse_args()
    run_pipeline(
        config_path=args.config,
        video_path=args.video,
        mode=args.mode,
        model=args.model,
        metadata=args.metadata,
    )


if __name__ == "__main__":
    main()
