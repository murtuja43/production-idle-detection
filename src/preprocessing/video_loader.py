"""Video reading and writing helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoProcessor:
    """Context-managed OpenCV video reader/writer."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        codec: str = "mp4v",
        output_fps: float | None = None,
        resize_enabled: bool = False,
        resize_width: int = 1280,
        resize_height: int = 720,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.codec = codec
        self.output_fps = output_fps
        self.resize_enabled = resize_enabled
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.capture: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.current_frame_index = 0
        self.timestamp_seconds = 0.0
        self.frame_count = 0
        self.fps = 0.0

    def __enter__(self) -> "VideoProcessor":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the input video and initialize output writer."""
        self.capture = cv2.VideoCapture(str(self.input_path))
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {self.input_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        input_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = self.resize_width if self.resize_enabled else input_width
        height = self.resize_height if self.resize_enabled else input_height
        output_fps = self.output_fps or self.fps

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            output_fps,
            (width, height),
        )
        if not self.writer.isOpened():
            raise ValueError(f"Could not open video writer: {self.output_path}")

    def read(self) -> np.ndarray | None:
        """Read the next frame, returning None at end-of-video."""
        if self.capture is None:
            raise RuntimeError("VideoProcessor must be opened before reading.")
        success, frame = self.capture.read()
        if not success:
            return None

        self.current_frame_index = int(
            self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        )
        self.timestamp_seconds = self.current_frame_index / max(self.fps, 1e-6)
        if self.resize_enabled:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        return frame

    def write(self, frame: np.ndarray) -> None:
        """Write a processed frame to the output video."""
        if self.writer is None:
            raise RuntimeError("VideoProcessor must be opened before writing.")
        self.writer.write(frame)

    def close(self) -> None:
        """Release OpenCV resources."""
        if self.capture is not None:
            self.capture.release()
        if self.writer is not None:
            self.writer.release()

