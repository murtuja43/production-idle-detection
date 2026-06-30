"""Command-line entry point for Phase 1 optical-flow idle detection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

from src.detection.idle_detector import IdleDetector
from src.optical_flow.dense_flow import DenseOpticalFlow
from src.preprocessing.roi import ZoneRegistry
from src.preprocessing.video_loader import VideoProcessor
from src.utils.config import load_config
from src.utils.csv_logger import CsvIdleLogger
from src.utils.logger import configure_logging
from src.visualization.overlay import OverlayRenderer


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
    return parser.parse_args()


def run_pipeline(config_path: str, video_path: str) -> None:
    """Run the complete optical-flow idle detection pipeline."""
    config = load_config(config_path)
    configure_logging(config.logging.level)
    logger = logging.getLogger(__name__)

    input_video = Path(video_path)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    zones = ZoneRegistry.from_config(config.zones)
    flow = DenseOpticalFlow(config.optical_flow)
    detector = IdleDetector.from_zones(zones)
    overlay = OverlayRenderer(config.visualization)

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

    logger.info("Starting idle detection for %s", input_video)
    logger.info("Writing processed video to %s", output_video)
    logger.info("Writing CSV log to %s", csv_path)

    previous_gray = None
    frame_count = processor.frame_count

    with processor, CsvIdleLogger(csv_path) as csv_logger:
        progress = tqdm(total=frame_count, desc="Processing frames", unit="frame")
        try:
            while True:
                frame = processor.read()
                if frame is None:
                    break

                frame_index = processor.current_frame_index
                timestamp_seconds = processor.timestamp_seconds
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                zone_results = []
                if previous_gray is not None:
                    flow_field = flow.compute(previous_gray, gray)
                    for zone in zones.enabled_zones:
                        motion_score = flow.motion_magnitude(
                            flow_field=flow_field,
                            gray_frame=gray,
                            zone=zone,
                        )
                        state = detector.update(
                            zone_name=zone.name,
                            motion_score=motion_score,
                            timestamp_seconds=timestamp_seconds,
                        )
                        zone_results.append((zone, motion_score, state))
                        csv_logger.write(
                            frame_index=frame_index,
                            timestamp_seconds=timestamp_seconds,
                            zone=zone,
                            motion_score=motion_score,
                            state=state,
                        )

                rendered = overlay.draw(frame, zone_results)
                processor.write(rendered)
                previous_gray = gray
                progress.update(1)
        finally:
            progress.close()

    logger.info("Finished processing %s", input_video)
    logger.info("Processed video saved to %s", output_video)
    logger.info("CSV log saved to %s", csv_path)


def main() -> None:
    """Run the CLI."""
    args = parse_args()
    run_pipeline(config_path=args.config, video_path=args.video)


if __name__ == "__main__":
    main()

