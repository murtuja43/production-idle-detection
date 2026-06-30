"""Video overlay rendering for idle detection results."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.idle_detector import ZoneDetectionState
from src.preprocessing.roi import Zone
from src.utils.config import VisualizationConfig


class OverlayRenderer:
    """Draw zone boxes, motion scores, and idle states on frames."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def draw(
        self,
        frame: np.ndarray,
        zone_results: list[tuple[Zone, ZoneDetectionState]],
    ) -> np.ndarray:
        """Render all zone results on a copy of the frame."""
        if not self.config.enabled:
            return frame

        output = frame.copy()
        for zone, state in zone_results:
            color = (0, 0, 255) if state.is_idle else (0, 180, 0)
            label = (
                f"{zone.name}: {'IDLE' if state.is_idle else 'ACTIVE'} "
                f"motion={state.motion_score:.2f}"
            )
            cv2.rectangle(
                output,
                (zone.x, zone.y),
                (zone.x2, zone.y2),
                color,
                self.config.line_thickness,
            )
            self._draw_label(output, label, zone.x, max(20, zone.y - 8), color)
        return output

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        text_size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.line_thickness,
        )
        width, height = text_size
        cv2.rectangle(
            frame,
            (x, y - height - baseline - 4),
            (x + width + 6, y + baseline),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            frame,
            text,
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            color,
            self.config.line_thickness,
            cv2.LINE_AA,
        )

