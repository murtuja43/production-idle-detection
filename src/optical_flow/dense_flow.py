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
    ) -> float:
        """Calculate a filtered mean motion magnitude for one zone."""
        roi_flow = zone.crop_array(flow_field)
        roi_gray = zone.crop_array(gray_frame)
        magnitude, _angle = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
        valid_mask = np.ones(magnitude.shape, dtype=np.uint8)

        roi_mask = self._get_roi_mask(zone)
        if roi_mask is not None:
            valid_mask = valid_mask & roi_mask

        if zone.is_cmus and zone.spark_filter.enabled:
            spark_mask = self._build_spark_mask(roi_gray, zone)
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
    def _build_spark_mask(gray_roi: np.ndarray, zone: Zone) -> np.ndarray:
        """Create a binary mask for bright CMUS welding sparks."""
        config = zone.spark_filter
        bright = (gray_roi >= config.brightness_threshold).astype(np.uint8)
        if config.dilate_iterations > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
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

