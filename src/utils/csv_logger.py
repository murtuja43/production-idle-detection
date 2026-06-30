"""CSV logging for per-zone idle detection results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from src.detection.idle_detector import ZoneDetectionState
from src.preprocessing.roi import Zone

if TYPE_CHECKING:  # avoid importing sklearn-backed modules in optical-flow mode
    from src.ml.inference import MlZoneResult


class CsvIdleLogger:
    """Context-managed CSV writer for idle detection events.

    In optical-flow mode (``include_ml=False``) the schema is exactly the Phase 1
    schema. In ML/combined mode (``include_ml=True``) extra columns expose the
    individual optical-flow and ML signals behind each final decision.
    """

    BASE_FIELDNAMES = [
        "frame_index",
        "timestamp_seconds",
        "zone",
        "motion_score",
        "motion_threshold",
        "is_motion_active",
        "is_idle",
        "idle_seconds",
        "sensitivity",
    ]
    ML_FIELDNAMES = [
        "mode",
        "optical_flow_is_idle",
        "ml_window_ready",
        "ml_is_anomaly",
        "ml_score",
    ]

    def __init__(self, output_path: Path, include_ml: bool = False) -> None:
        self.output_path = output_path
        self.include_ml = include_ml
        self.fieldnames = list(self.BASE_FIELDNAMES)
        if include_ml:
            self.fieldnames += self.ML_FIELDNAMES
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "CsvIdleLogger":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._file is not None:
            self._file.close()

    def write(
        self,
        frame_index: int,
        timestamp_seconds: float,
        zone: Zone,
        motion_score: float,
        state: ZoneDetectionState,
        *,
        mode: str | None = None,
        optical_flow_idle: bool | None = None,
        ml_result: "MlZoneResult | None" = None,
    ) -> None:
        """Write one per-zone frame result.

        ``state`` carries the final decision for the active mode. The keyword-only
        ``mode``/``optical_flow_idle``/``ml_result`` arguments populate the extra
        ML columns and are ignored unless ``include_ml`` is set.
        """
        if self._writer is None:
            raise RuntimeError("CsvIdleLogger must be opened before writing.")

        row: dict[str, object] = {
            "frame_index": frame_index,
            "timestamp_seconds": f"{timestamp_seconds:.3f}",
            "zone": zone.name,
            "motion_score": f"{motion_score:.6f}",
            "motion_threshold": f"{zone.motion_threshold:.6f}",
            "is_motion_active": int(state.is_motion_active),
            "is_idle": int(state.is_idle),
            "idle_seconds": f"{state.idle_seconds:.3f}",
            "sensitivity": f"{zone.sensitivity:.3f}",
        }

        if self.include_ml:
            row["mode"] = mode if mode is not None else ""
            row["optical_flow_is_idle"] = (
                int(optical_flow_idle) if optical_flow_idle is not None else ""
            )
            if ml_result is not None and ml_result.window_ready:
                row["ml_window_ready"] = 1
                row["ml_is_anomaly"] = int(ml_result.is_anomaly)
                row["ml_score"] = f"{ml_result.score:.6f}"
            else:
                row["ml_window_ready"] = 0
                row["ml_is_anomaly"] = ""
                row["ml_score"] = ""

        self._writer.writerow(row)
