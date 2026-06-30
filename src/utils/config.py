"""Configuration loading and typed config objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResizeConfig:
    """Video resize settings."""

    enabled: bool
    width: int
    height: int


@dataclass(frozen=True)
class VideoConfig:
    """Video input/output settings."""

    output_dir: str
    output_fps: float | None
    codec: str
    resize: ResizeConfig


@dataclass(frozen=True)
class LoggingConfig:
    """Application logging and CSV settings."""

    level: str
    csv_dir: str
    csv_filename: str


@dataclass(frozen=True)
class OpticalFlowConfig:
    """OpenCV Farneback dense optical-flow parameters."""

    pyr_scale: float
    levels: int
    winsize: int
    iterations: int
    poly_n: int
    poly_sigma: float
    flags: int


@dataclass(frozen=True)
class VisualizationConfig:
    """Overlay rendering settings."""

    enabled: bool
    draw_motion_heatmap: bool
    font_scale: float
    line_thickness: int


@dataclass(frozen=True)
class RoiConfig:
    """Rectangular region-of-interest settings."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class SparkFilterConfig:
    """CMUS spark filtering settings."""

    enabled: bool = False
    brightness_threshold: int = 230
    min_component_area: int = 4
    dilate_iterations: int = 1


@dataclass(frozen=True)
class ZoneConfig:
    """Per-production-zone detection settings."""

    name: str
    enabled: bool
    roi: RoiConfig
    motion_threshold: float
    idle_duration_seconds: float
    sensitivity: float
    mask_path: str | None
    spark_filter: SparkFilterConfig


@dataclass(frozen=True)
class AppConfig:
    """Top-level application config."""

    video: VideoConfig
    logging: LoggingConfig
    optical_flow: OpticalFlowConfig
    visualization: VisualizationConfig
    zones: dict[str, ZoneConfig]


REQUIRED_ZONES = ("CMUS", "COP", "COK", "CSK", "CSLT")


def _require_mapping(data: Any, key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def _parse_resize(data: dict[str, Any]) -> ResizeConfig:
    resize = _require_mapping(data, "resize")
    return ResizeConfig(
        enabled=bool(resize.get("enabled", False)),
        width=int(resize.get("width", 1280)),
        height=int(resize.get("height", 720)),
    )


def _parse_video(data: dict[str, Any]) -> VideoConfig:
    output_fps = data.get("output_fps")
    return VideoConfig(
        output_dir=str(data.get("output_dir", "data/processed")),
        output_fps=None if output_fps is None else float(output_fps),
        codec=str(data.get("codec", "mp4v")),
        resize=_parse_resize(data),
    )


def _parse_logging(data: dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(
        level=str(data.get("level", "INFO")),
        csv_dir=str(data.get("csv_dir", "outputs")),
        csv_filename=str(data.get("csv_filename", "idle_detection_log.csv")),
    )


def _parse_optical_flow(data: dict[str, Any]) -> OpticalFlowConfig:
    return OpticalFlowConfig(
        pyr_scale=float(data.get("pyr_scale", 0.5)),
        levels=int(data.get("levels", 3)),
        winsize=int(data.get("winsize", 15)),
        iterations=int(data.get("iterations", 3)),
        poly_n=int(data.get("poly_n", 5)),
        poly_sigma=float(data.get("poly_sigma", 1.2)),
        flags=int(data.get("flags", 0)),
    )


def _parse_visualization(data: dict[str, Any]) -> VisualizationConfig:
    return VisualizationConfig(
        enabled=bool(data.get("enabled", True)),
        draw_motion_heatmap=bool(data.get("draw_motion_heatmap", False)),
        font_scale=float(data.get("font_scale", 0.55)),
        line_thickness=int(data.get("line_thickness", 2)),
    )


def _parse_roi(data: dict[str, Any], zone_name: str) -> RoiConfig:
    roi = _require_mapping(data, "roi")
    parsed = RoiConfig(
        x=int(roi.get("x", 0)),
        y=int(roi.get("y", 0)),
        width=int(roi.get("width", 0)),
        height=int(roi.get("height", 0)),
    )
    if parsed.width <= 0 or parsed.height <= 0:
        raise ValueError(f"Zone '{zone_name}' ROI width and height must be > 0.")
    if parsed.x < 0 or parsed.y < 0:
        raise ValueError(f"Zone '{zone_name}' ROI x/y must be >= 0.")
    return parsed


def _parse_spark_filter(data: dict[str, Any]) -> SparkFilterConfig:
    spark_data = data.get("spark_filter", {})
    if not isinstance(spark_data, dict):
        raise ValueError("spark_filter must be a mapping when provided.")
    return SparkFilterConfig(
        enabled=bool(spark_data.get("enabled", False)),
        brightness_threshold=int(spark_data.get("brightness_threshold", 230)),
        min_component_area=int(spark_data.get("min_component_area", 4)),
        dilate_iterations=int(spark_data.get("dilate_iterations", 1)),
    )


def _parse_zones(data: dict[str, Any]) -> dict[str, ZoneConfig]:
    zones: dict[str, ZoneConfig] = {}
    for zone_name in REQUIRED_ZONES:
        raw_zone = data.get(zone_name)
        if not isinstance(raw_zone, dict):
            raise ValueError(f"Missing required zone configuration: {zone_name}")
        zones[zone_name] = ZoneConfig(
            name=zone_name,
            enabled=bool(raw_zone.get("enabled", True)),
            roi=_parse_roi(raw_zone, zone_name),
            motion_threshold=float(raw_zone.get("motion_threshold", 1.0)),
            idle_duration_seconds=float(
                raw_zone.get("idle_duration_seconds", 5.0)
            ),
            sensitivity=float(raw_zone.get("sensitivity", 1.0)),
            mask_path=raw_zone.get("mask_path"),
            spark_filter=_parse_spark_filter(raw_zone),
        )
        if zones[zone_name].motion_threshold <= 0:
            raise ValueError(f"Zone '{zone_name}' motion_threshold must be > 0.")
        if zones[zone_name].idle_duration_seconds < 0:
            raise ValueError(
                f"Zone '{zone_name}' idle_duration_seconds must be >= 0."
            )
        if zones[zone_name].sensitivity <= 0:
            raise ValueError(f"Zone '{zone_name}' sensitivity must be > 0.")
    return zones


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an application YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return AppConfig(
        video=_parse_video(_require_mapping(raw, "video")),
        logging=_parse_logging(_require_mapping(raw, "logging")),
        optical_flow=_parse_optical_flow(_require_mapping(raw, "optical_flow")),
        visualization=_parse_visualization(
            _require_mapping(raw, "visualization")
        ),
        zones=_parse_zones(_require_mapping(raw, "zones")),
    )

