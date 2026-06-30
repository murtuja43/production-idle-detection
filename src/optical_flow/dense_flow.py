"""Dense optical-flow computation and per-zone motion measurement."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.roi import Zone
from src.utils.config import OpticalFlowConfig


class DenseOpticalFlow:
    """OpenCV Farneback dense optical-flow wrapper."""

    def __init__(self, config: OpticalFlowConfig) -> None:
        self.config = config
        self._mask_cache: dict[str, np.ndarray | None] = {}

    def compute(self, previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
        """Compute Farneback dense optical flow between two grayscale frames."""
        return cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            self.config.pyr_scale,
            self.config.levels,
            self.config.winsize,
            self.config.iterations,
            self.config.poly_n,
            self.config.poly_sigma,
            self.config.flags,
        )

    def motion_magnitude(
        self,
        flow_field: np.ndarray,
        gray_frame: np.ndarray,
        zone: Zone,
        color_frame: np.ndarray | None = None,
    ) -> float:
        """Calculate a filtered mean motion magnitude for one zone.

        ``color_frame`` (BGR) is optional and only used when a zone's spark filter
        enables saturation-based glare gating; everything else relies solely on
        the grayscale frame, so callers may omit it.
        """
        roi_flow = zone.crop_array(flow_field)
        roi_gray = zone.crop_array(gray_frame)
        magnitude = np.hypot(roi_flow[..., 0], roi_flow[..., 1])
        valid_mask = np.ones(magnitude.shape, dtype=np.uint8)

        roi_mask = self._get_roi_mask(zone)
        if roi_mask is not None:
            valid_mask = valid_mask & roi_mask

        if zone.is_cmus and zone.spark_filter.enabled:
            roi_color = (
                zone.crop_array(color_frame) if color_frame is not None else None
            )
            spark_mask = self._build_spark_mask(roi_gray, roi_color, zone)
            valid_mask = valid_mask & (1 - spark_mask)

        valid_values = magnitude[valid_mask.astype(bool)]
        if valid_values.size == 0:
            return 0.0

        return float(np.mean(valid_values) * zone.sensitivity)

    def _get_roi_mask(self, zone: Zone) -> np.ndarray | None:
        if zone.name not in self._mask_cache:
            self._mask_cache[zone.name] = zone.load_mask()
        return self._mask_cache[zone.name]

    @staticmethod
    def _build_spark_mask(
        gray_roi: np.ndarray,
        color_roi: np.ndarray | None,
        zone: Zone,
    ) -> np.ndarray:
        """Create a binary mask (1 = spark/glare) for the CMUS welding area."""
        config = zone.spark_filter
        bright = (gray_roi >= config.brightness_threshold).astype(np.uint8)

        # Optional colour gate: only treat bright pixels that are also
        # low-saturation (white/blue-white glare) as sparks.
        if config.saturation_threshold is not None and color_roi is not None:
            hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
            low_saturation = (hsv[..., 1] <= config.saturation_threshold).astype(
                np.uint8
            )
            bright = bright & low_saturation

        if config.dilate_iterations > 0:
            kernel = np.ones((config.kernel_size, config.kernel_size), dtype=np.uint8)
            bright = cv2.dilate(
                bright,
                kernel,
                iterations=config.dilate_iterations,
            )

        if config.min_component_area <= 1:
            return bright

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            bright,
            connectivity=8,
        )
        filtered = np.zeros_like(bright)
        for component_index in range(1, component_count):
            area = stats[component_index, cv2.CC_STAT_AREA]
            if area >= config.min_component_area:
                filtered[labels == component_index] = 1
        return filtered

