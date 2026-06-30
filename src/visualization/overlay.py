"""Video overlay rendering for idle detection results."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocessing.roi import Zone
from src.utils.config import VisualizationConfig

_IDLE_COLOR = (0, 0, 255)
_ACTIVE_COLOR = (0, 180, 0)
_BANNER_COLOR = (240, 240, 240)


@dataclass(frozen=True)
class OverlayZone:
    """Per-zone data needed to render an overlay (decoupled from detection)."""

    zone: Zone
    is_idle: bool
    motion_score: float
    threshold: float
    idle_seconds: float
    anomaly_score: float | None = None


class OverlayRenderer:
    """Draw zone boxes, motion/idle details, and the active mode on frames."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def draw(
        self,
        frame: np.ndarray,
        zones: list[OverlayZone],
        mode: str,
    ) -> np.ndarray:
        """Render all zone overlays and a mode banner on a copy of the frame."""
        if not self.config.enabled:
            return frame

        output = frame.copy()
        for item in zones:
            color = _IDLE_COLOR if item.is_idle else _ACTIVE_COLOR
            zone = item.zone
            cv2.rectangle(
                output,
                (zone.x, zone.y),
                (zone.x2, zone.y2),
                color,
                self.config.line_thickness,
            )
            lines = [
                f"{zone.name}  {'IDLE' if item.is_idle else 'ACTIVE'}",
                f"motion {item.motion_score:.2f} / thr {item.threshold:.2f}",
                f"idle {item.idle_seconds:.1f}s",
            ]
            if item.anomaly_score is not None:
                lines.append(f"anomaly {item.anomaly_score:+.3f}")
            self._draw_multiline(output, lines, zone.x, zone.y, color)

        self._draw_banner(output, f"MODE: {mode.upper()}")
        return output

    def _draw_multiline(
        self,
        frame: np.ndarray,
        lines: list[str],
        x: int,
        y_top: int,
        color: tuple[int, int, int],
    ) -> None:
        scale = self.config.font_scale
        thickness = max(1, self.config.line_thickness - 1)
        (_, text_height), baseline = cv2.getTextSize(
            "Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        line_height = text_height + baseline + 4
        # Stack labels just below the top edge of the ROI.
        y = max(line_height, y_top) + 2
        for line in lines:
            (width, _), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )
            cv2.rectangle(
                frame,
                (x, y - text_height - baseline),
                (x + width + 6, y + 2),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                frame,
                line,
                (x + 3, y - baseline + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += line_height

    def _draw_banner(self, frame: np.ndarray, text: str) -> None:
        scale = self.config.font_scale + 0.1
        thickness = self.config.line_thickness
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        cv2.rectangle(frame, (0, 0), (width + 12, height + baseline + 10), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (6, height + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            _BANNER_COLOR,
            thickness,
            cv2.LINE_AA,
        )
