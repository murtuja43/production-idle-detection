"""Tests for the reusable motion-processing engine."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.optical_flow.dense_flow import DenseOpticalFlow
from src.pipeline.motion_pipeline import MotionPipeline
from src.preprocessing.roi import Zone
from src.preprocessing.video_loader import VideoProcessor
from src.utils.config import OpticalFlowConfig, SparkFilterConfig


def _flow() -> DenseOpticalFlow:
    return DenseOpticalFlow(
        OpticalFlowConfig(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    )


def _zone(name: str = "COP", width: int = 120, height: int = 120) -> Zone:
    return Zone(
        name=name,
        enabled=True,
        x=0,
        y=0,
        width=width,
        height=height,
        motion_threshold=1.0,
        idle_duration_seconds=3.0,
        sensitivity=1.0,
        mask_path=None,
        spark_filter=SparkFilterConfig(),
    )


def _write_video(path: Path, frames: int = 6, size: tuple[int, int] = (320, 240)) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic test video.")
    try:
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 10 + index * 6
            cv2.rectangle(frame, (x, 50), (x + 30, 90), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


class MotionPipelineTest(unittest.TestCase):
    """Validate the shared per-frame motion engine."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="motion-pipeline-test-"))
        self.video_path = self.temp_dir / "sample.mp4"
        _write_video(self.video_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_first_frame_has_no_motion_then_indices_are_zero_based(self) -> None:
        pipeline = MotionPipeline(_flow(), [_zone()])
        with VideoProcessor(self.video_path) as processor:
            results = list(pipeline.iter_motion(processor))

        self.assertGreater(len(results), 1)
        first = results[0]
        self.assertEqual(first.frame_index, 0)
        self.assertEqual(first.timestamp_seconds, 0.0)
        self.assertFalse(first.has_motion)
        self.assertEqual(first.zone_motions, [])

        second = results[1]
        self.assertEqual(second.frame_index, 1)
        self.assertAlmostEqual(second.timestamp_seconds, 0.1, places=6)
        self.assertTrue(second.has_motion)
        self.assertEqual(len(second.zone_motions), 1)

    def test_motion_is_detected_for_moving_object(self) -> None:
        pipeline = MotionPipeline(_flow(), [_zone()])
        with VideoProcessor(self.video_path) as processor:
            scores = [
                frame.zone_motions[0].motion_score
                for frame in pipeline.iter_motion(processor)
                if frame.has_motion
            ]

        self.assertTrue(scores)
        self.assertGreater(max(scores), 0.0)

    def test_read_only_processor_does_not_require_output(self) -> None:
        with VideoProcessor(self.video_path) as processor:
            self.assertIsNone(processor.writer)
            self.assertEqual(processor.width, 320)
            self.assertEqual(processor.height, 240)
            with self.assertRaises(RuntimeError):
                processor.write(np.zeros((240, 320, 3), dtype=np.uint8))

    def test_out_of_bounds_roi_is_rejected(self) -> None:
        oversized = _zone(width=400, height=400)
        pipeline = MotionPipeline(_flow(), [oversized])
        with VideoProcessor(self.video_path) as processor:
            with self.assertRaises(ValueError):
                list(pipeline.iter_motion(processor))


if __name__ == "__main__":
    unittest.main()
