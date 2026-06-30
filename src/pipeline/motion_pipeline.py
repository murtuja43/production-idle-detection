"""Per-frame motion-processing engine shared across detection modes.

This module owns the single canonical "read frame -> grayscale -> dense optical
flow -> per-zone motion magnitude" loop. Phase 1 optical-flow detection consumes
it directly, and Phase 2 (feature extraction, dataset generation, and ML
inference) reuses the very same engine so that every mode measures motion
identically rather than re-implementing the loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from src.optical_flow.dense_flow import DenseOpticalFlow
from src.preprocessing.roi import Zone
from src.preprocessing.video_loader import VideoProcessor


@dataclass(frozen=True)
class ZoneMotion:
    """Motion measurement for a single zone within a single frame."""

    zone: Zone
    motion_score: float


@dataclass(frozen=True)
class FrameMotion:
    """All per-zone motion measurements for one processed frame.

    The first frame of a video has no predecessor, so optical flow cannot be
    computed for it; in that case ``zone_motions`` is empty and ``has_motion``
    is ``False``.
    """

    frame_index: int
    timestamp_seconds: float
    frame: np.ndarray
    gray: np.ndarray
    zone_motions: list[ZoneMotion]

    @property
    def has_motion(self) -> bool:
        """Whether optical-flow motion was computed for this frame."""
        return bool(self.zone_motions)


class MotionPipeline:
    """Stream per-frame, per-zone motion scores from a video.

    The pipeline is stateless across runs: each call to :meth:`iter_motion`
    drives a freshly opened :class:`VideoProcessor` and tracks its own
    previous-frame state, so the same instance can process several videos.
    """

    def __init__(self, flow: DenseOpticalFlow, zones: list[Zone]) -> None:
        self._flow = flow
        self._zones = list(zones)

    @property
    def zones(self) -> list[Zone]:
        """Zones measured by this pipeline."""
        return self._zones

    def iter_motion(self, processor: VideoProcessor) -> Iterator[FrameMotion]:
        """Yield a :class:`FrameMotion` for every frame read from ``processor``.

        ``processor`` must already be opened (its frame dimensions are used to
        validate each zone's ROI before processing begins).
        """
        for zone in self._zones:
            zone.ensure_within_frame(processor.width, processor.height)

        previous_gray: np.ndarray | None = None
        while True:
            frame = processor.read()
            if frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            zone_motions: list[ZoneMotion] = []
            if previous_gray is not None:
                flow_field = self._flow.compute(previous_gray, gray)
                for zone in self._zones:
                    score = self._flow.motion_magnitude(
                        flow_field=flow_field,
                        gray_frame=gray,
                        zone=zone,
                        color_frame=frame,
                    )
                    zone_motions.append(ZoneMotion(zone=zone, motion_score=score))

            yield FrameMotion(
                frame_index=processor.current_frame_index,
                timestamp_seconds=processor.timestamp_seconds,
                frame=frame,
                gray=gray,
                zone_motions=zone_motions,
            )
            previous_gray = gray
