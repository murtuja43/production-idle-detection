"""Production-zone ROI models and mask handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import RoiConfig, SparkFilterConfig, ZoneConfig


@dataclass(frozen=True)
class Zone:
    """Runtime production-zone definition."""

    name: str
    enabled: bool
    x: int
    y: int
    width: int
    height: int
    motion_threshold: float
    idle_duration_seconds: float
    sensitivity: float
    mask_path: str | None
    spark_filter: SparkFilterConfig

    @classmethod
    def from_config(cls, config: ZoneConfig) -> "Zone":
        """Create a runtime zone from config."""
        roi: RoiConfig = config.roi
        return cls(
            name=config.name,
            enabled=config.enabled,
            x=roi.x,
            y=roi.y,
            width=roi.width,
            height=roi.height,
            motion_threshold=config.motion_threshold,
            idle_duration_seconds=config.idle_duration_seconds,
            sensitivity=config.sensitivity,
            mask_path=config.mask_path,
            spark_filter=config.spark_filter,
        )

    @property
    def x2(self) -> int:
        """Right ROI coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom ROI coordinate."""
        return self.y + self.height

    @property
    def is_cmus(self) -> bool:
        """Return whether this zone is CMUS."""
        return self.name.upper() == "CMUS"

    def ensure_within_frame(self, width: int, height: int) -> None:
        """Raise if this ROI extends past a frame of the given size.

        numpy slicing silently clamps an out-of-bounds ROI to a smaller crop,
        which would both score the wrong region and break mask broadcasting, so
        fail fast with a clear message instead.
        """
        if self.x2 > width or self.y2 > height:
            raise ValueError(
                f"Zone '{self.name}' ROI "
                f"({self.x}, {self.y}, {self.width}, {self.height}) "
                f"exceeds frame bounds {width}x{height}."
            )

    def crop_array(self, array: np.ndarray) -> np.ndarray:
        """Return the ROI crop from an image-like array."""
        return array[self.y : self.y2, self.x : self.x2]

    def load_mask(self) -> np.ndarray | None:
        """Load an optional binary ROI mask for this zone."""
        if not self.mask_path:
            return None
        path = Path(self.mask_path)
        if not path.exists():
            raise FileNotFoundError(f"Mask not found for zone {self.name}: {path}")

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask for zone {self.name}: {path}")
        if mask.shape != (self.height, self.width):
            # Nearest-neighbour keeps the mask strictly binary (no interpolated
            # edge values) when the source image size differs from the ROI.
            mask = cv2.resize(
                mask,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )
        return (mask > 0).astype(np.uint8)


class ZoneRegistry:
    """Container for configured production zones."""

    def __init__(self, zones: list[Zone]) -> None:
        self.zones = zones

    @classmethod
    def from_config(cls, configs: dict[str, ZoneConfig]) -> "ZoneRegistry":
        """Build a registry from config objects."""
        return cls([Zone.from_config(config) for config in configs.values()])

    @property
    def enabled_zones(self) -> list[Zone]:
        """Return enabled zones."""
        return [zone for zone in self.zones if zone.enabled]

    def names(self) -> list[str]:
        """Return all configured zone names."""
        return [zone.name for zone in self.zones]

