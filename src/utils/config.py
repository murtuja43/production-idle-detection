"""Configuration loading and typed config objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.features.extractor import AVAILABLE_FEATURES


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
    """Application logging, CSV, and report settings."""

    level: str
    csv_dir: str
    csv_filename: str
    report_csv_filename: str
    report_json_filename: str
    report_chart: bool
    report_chart_filename: str


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
    """CMUS spark/glare filtering settings.

    A pixel is treated as spark/glare when it is bright (>= ``brightness_threshold``)
    and, when ``saturation_threshold`` is set, also low-saturation (<= that value),
    which targets the white/blue-white glare of welding without masking genuinely
    coloured moving parts. Such pixels are excluded from motion scoring.
    """

    enabled: bool = False
    brightness_threshold: int = 230
    min_component_area: int = 4
    dilate_iterations: int = 1
    kernel_size: int = 3
    saturation_threshold: int | None = None


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
class DetectionConfig:
    """Detection mode selection."""

    mode: str  # one of VALID_MODES


@dataclass(frozen=True)
class FeatureConfig:
    """Sliding-window motion feature settings shared by training and inference."""

    window_size: int
    step: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class CombineConfig:
    """How combined mode fuses optical-flow and ML signals."""

    strategy: str  # one of VALID_COMBINE_STRATEGIES


@dataclass(frozen=True)
class MlConfig:
    """ML inference settings (model location and combine policy)."""

    model_path: str
    metadata_path: str
    combine: CombineConfig


@dataclass(frozen=True)
class TrainingConfig:
    """Isolation Forest training hyperparameters and output paths."""

    contamination: float | str
    random_state: int
    n_estimators: int
    max_samples: int | float | str
    feature_csv: str
    model_output: str
    metadata_output: str


@dataclass(frozen=True)
class AppConfig:
    """Top-level application config."""

    video: VideoConfig
    logging: LoggingConfig
    optical_flow: OpticalFlowConfig
    visualization: VisualizationConfig
    zones: dict[str, ZoneConfig]
    detection: DetectionConfig
    features: FeatureConfig
    ml: MlConfig
    training: TrainingConfig


REQUIRED_ZONES = ("CMUS", "COP", "COK", "CSK", "CSLT")
VALID_MODES = ("optical_flow", "ml", "combined")
VALID_COMBINE_STRATEGIES = ("and", "or")
# Default feature set; the canonical registry of available features lives in
# src.features.extractor, which validates names at extraction time.
DEFAULT_FEATURES = ("mean", "std", "max", "min", "active_ratio", "mean_delta")
DEFAULT_MODEL_PATH = "data/models/isolation_forest.joblib"
DEFAULT_METADATA_PATH = "data/models/isolation_forest.metadata.json"


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
        report_csv_filename=str(data.get("report_csv_filename", "idle_report.csv")),
        report_json_filename=str(
            data.get("report_json_filename", "idle_report.json")
        ),
        report_chart=bool(data.get("report_chart", True)),
        report_chart_filename=str(
            data.get("report_chart_filename", "idle_report.png")
        ),
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


def _parse_spark_filter(data: dict[str, Any], zone_name: str) -> SparkFilterConfig:
    spark_data = data.get("spark_filter", {})
    if not isinstance(spark_data, dict):
        raise ValueError(
            f"Zone '{zone_name}' spark_filter must be a mapping when provided."
        )
    brightness = int(spark_data.get("brightness_threshold", 230))
    if not 0 <= brightness <= 255:
        raise ValueError(
            f"Zone '{zone_name}' spark_filter.brightness_threshold must be 0-255."
        )
    kernel_size = int(spark_data.get("kernel_size", 3))
    if kernel_size < 1:
        raise ValueError(
            f"Zone '{zone_name}' spark_filter.kernel_size must be >= 1."
        )
    saturation_raw = spark_data.get("saturation_threshold")
    saturation = None if saturation_raw is None else int(saturation_raw)
    if saturation is not None and not 0 <= saturation <= 255:
        raise ValueError(
            f"Zone '{zone_name}' spark_filter.saturation_threshold must be 0-255."
        )
    return SparkFilterConfig(
        enabled=bool(spark_data.get("enabled", False)),
        brightness_threshold=brightness,
        min_component_area=int(spark_data.get("min_component_area", 4)),
        dilate_iterations=int(spark_data.get("dilate_iterations", 1)),
        kernel_size=kernel_size,
        saturation_threshold=saturation,
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
            spark_filter=_parse_spark_filter(raw_zone, zone_name),
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


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a sub-mapping if present, otherwise an empty mapping.

    Lets the new Phase 2 sections (detection/features/ml/training) be omitted so
    older Phase 1 configs continue to load with sensible defaults.
    """
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping when provided.")
    return value


def _parse_detection(data: dict[str, Any]) -> DetectionConfig:
    mode = str(data.get("mode", "optical_flow")).lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"detection.mode must be one of {list(VALID_MODES)}; got '{mode}'."
        )
    return DetectionConfig(mode=mode)


def _parse_features(data: dict[str, Any]) -> FeatureConfig:
    window_size = int(data.get("window_size", 30))
    step = int(data.get("step", 15))
    raw_features = data.get("features", list(DEFAULT_FEATURES))
    if not isinstance(raw_features, (list, tuple)) or not raw_features:
        raise ValueError("features.features must be a non-empty list of names.")
    features = tuple(str(name) for name in raw_features)
    unknown = [name for name in features if name not in AVAILABLE_FEATURES]
    if unknown:
        raise ValueError(
            f"features.features contains unknown feature(s): {unknown}. "
            f"Available: {sorted(AVAILABLE_FEATURES)}."
        )
    if window_size < 2:
        raise ValueError("features.window_size must be >= 2.")
    if step < 1:
        raise ValueError("features.step must be >= 1.")
    return FeatureConfig(window_size=window_size, step=step, features=features)


def _parse_contamination(value: Any) -> float | str:
    if value is None:
        return "auto"
    if isinstance(value, str):
        if value.lower() == "auto":
            return "auto"
        value = float(value)
    contamination = float(value)
    if not 0.0 < contamination <= 0.5:
        raise ValueError(
            "training.contamination must be 'auto' or a float in (0, 0.5]."
        )
    return contamination


def _parse_max_samples(value: Any) -> int | float | str:
    if value is None:
        return "auto"
    if isinstance(value, str):
        return "auto" if value.lower() == "auto" else float(value)
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise ValueError("training.max_samples must be 'auto', an int, or a float.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("training.max_samples (int) must be > 0.")
        return int(value)
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise ValueError("training.max_samples (float) must be in (0, 1].")
    return number


def _parse_combine(data: dict[str, Any]) -> CombineConfig:
    strategy = str(data.get("strategy", "and")).lower()
    if strategy not in VALID_COMBINE_STRATEGIES:
        raise ValueError(
            f"ml.combine.strategy must be one of {list(VALID_COMBINE_STRATEGIES)}; "
            f"got '{strategy}'."
        )
    return CombineConfig(strategy=strategy)


def _parse_ml(data: dict[str, Any]) -> MlConfig:
    return MlConfig(
        model_path=str(data.get("model_path", DEFAULT_MODEL_PATH)),
        metadata_path=str(data.get("metadata_path", DEFAULT_METADATA_PATH)),
        combine=_parse_combine(_optional_mapping(data, "combine")),
    )


def _parse_training(data: dict[str, Any], ml: MlConfig) -> TrainingConfig:
    n_estimators = int(data.get("n_estimators", 100))
    random_state = int(data.get("random_state", 42))
    if n_estimators <= 0:
        raise ValueError("training.n_estimators must be > 0.")
    return TrainingConfig(
        contamination=_parse_contamination(data.get("contamination")),
        random_state=random_state,
        n_estimators=n_estimators,
        max_samples=_parse_max_samples(data.get("max_samples")),
        feature_csv=str(data.get("feature_csv", "data/processed/features.csv")),
        model_output=str(data.get("model_output", ml.model_path)),
        metadata_output=str(data.get("metadata_output", ml.metadata_path)),
    )


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an application YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    ml = _parse_ml(_optional_mapping(raw, "ml"))
    return AppConfig(
        video=_parse_video(_require_mapping(raw, "video")),
        logging=_parse_logging(_require_mapping(raw, "logging")),
        optical_flow=_parse_optical_flow(_require_mapping(raw, "optical_flow")),
        visualization=_parse_visualization(
            _require_mapping(raw, "visualization")
        ),
        zones=_parse_zones(_require_mapping(raw, "zones")),
        detection=_parse_detection(_optional_mapping(raw, "detection")),
        features=_parse_features(_optional_mapping(raw, "features")),
        ml=ml,
        training=_parse_training(_optional_mapping(raw, "training"), ml),
    )

