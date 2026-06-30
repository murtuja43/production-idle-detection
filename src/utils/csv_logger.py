"""CSV logging for per-zone idle detection results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from src.detection.idle_detector import ZoneDetectionState
from src.preprocessing.roi import Zone


class CsvIdleLogger:
    """Context-managed CSV writer for idle detection events."""

    FIELDNAMES = [
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

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "CsvIdleLogger":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
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
    ) -> None:
        """Write one per-zone frame result."""
        if self._writer is None:
            raise RuntimeError("CsvIdleLogger must be opened before writing.")

        self._writer.writerow(
            {
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
        )

