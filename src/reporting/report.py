"""Per-zone idle-detection reporting.

Accumulates statistics while a video is processed and writes a per-zone summary
to CSV and JSON (and, optionally, a bar chart). The aggregator takes primitive
values rather than detection objects, so reporting stays decoupled from the
detection/ML layers and is trivial to unit-test.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class _ZoneAccumulator:
    """Mutable per-zone counters used while processing frames."""

    frames: int = 0
    idle_frames: int = 0
    motion_sum: float = 0.0
    idle_events: int = 0
    anomaly_count: int = 0
    _prev_idle: bool = field(default=False, repr=False)

    def update(self, motion_score: float, is_idle: bool, is_anomaly: bool) -> None:
        self.frames += 1
        self.motion_sum += motion_score
        if is_idle:
            self.idle_frames += 1
            if not self._prev_idle:
                self.idle_events += 1
        self._prev_idle = is_idle
        if is_anomaly:
            self.anomaly_count += 1


@dataclass(frozen=True)
class ZoneReport:
    """Summary statistics for a single zone."""

    zone: str
    frames: int
    total_seconds: float
    idle_seconds: float
    active_seconds: float
    idle_events: int
    average_motion: float
    anomaly_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Report:
    """Full processing report: a summary plus per-zone statistics."""

    video: str
    mode: str
    fps: float
    frames_evaluated: int
    generated_at: str
    zones: list[ZoneReport]

    def summary(self) -> dict[str, object]:
        """Top-level aggregate summary across all zones."""
        return {
            "video": self.video,
            "mode": self.mode,
            "fps": round(self.fps, 3),
            "frames_evaluated": self.frames_evaluated,
            "generated_at": self.generated_at,
            "zone_count": len(self.zones),
            "total_idle_seconds": round(
                sum(zone.idle_seconds for zone in self.zones), 3
            ),
            "total_idle_events": sum(zone.idle_events for zone in self.zones),
            "total_anomalies": sum(zone.anomaly_count for zone in self.zones),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "zones": [zone.to_dict() for zone in self.zones],
        }

    def save_json(self, path: str | Path) -> None:
        """Write the report to a JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)

    def save_csv(self, path: str | Path) -> None:
        """Write per-zone statistics to a CSV file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "zone", "frames", "total_seconds", "idle_seconds", "active_seconds",
            "idle_events", "average_motion", "anomaly_count",
        ]
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for zone in self.zones:
                writer.writerow(zone.to_dict())

    def save_chart(self, path: str | Path) -> bool:
        """Write a per-zone idle-time bar chart. Returns False if unavailable.

        Charting failures (e.g. a missing backend) are logged and swallowed so
        they never abort a processing run.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as error:  # pragma: no cover - environment dependent
            logger.warning("Skipping report chart (matplotlib unavailable): %s", error)
            return False

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        names = [zone.zone for zone in self.zones]
        idle = [zone.idle_seconds for zone in self.zones]
        active = [zone.active_seconds for zone in self.zones]

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(names, active, label="Active (s)", color="#2e7d32")
        axis.bar(names, idle, bottom=active, label="Idle (s)", color="#c62828")
        axis.set_ylabel("Seconds")
        axis.set_title(f"Idle vs active time per zone (mode={self.mode})")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=120)
        plt.close(figure)
        return True


class ReportAggregator:
    """Accumulate per-zone statistics across processed frames."""

    def __init__(self, zone_names: list[str], mode: str, fps: float) -> None:
        self._mode = mode
        self._fps = fps if fps > 0 else 30.0
        self._zones = {name: _ZoneAccumulator() for name in zone_names}

    def update(
        self,
        zone_name: str,
        motion_score: float,
        is_idle: bool,
        is_anomaly: bool = False,
    ) -> None:
        """Record one per-zone frame result."""
        accumulator = self._zones.get(zone_name)
        if accumulator is None:
            accumulator = _ZoneAccumulator()
            self._zones[zone_name] = accumulator
        accumulator.update(motion_score, is_idle, is_anomaly)

    def build(self, video: str) -> Report:
        """Finalize accumulators into an immutable :class:`Report`."""
        period = 1.0 / self._fps
        zone_reports: list[ZoneReport] = []
        frames_evaluated = 0
        for name, acc in self._zones.items():
            frames_evaluated = max(frames_evaluated, acc.frames)
            average_motion = acc.motion_sum / acc.frames if acc.frames else 0.0
            zone_reports.append(
                ZoneReport(
                    zone=name,
                    frames=acc.frames,
                    total_seconds=round(acc.frames * period, 3),
                    idle_seconds=round(acc.idle_frames * period, 3),
                    active_seconds=round((acc.frames - acc.idle_frames) * period, 3),
                    idle_events=acc.idle_events,
                    average_motion=round(average_motion, 6),
                    anomaly_count=acc.anomaly_count,
                )
            )
        return Report(
            video=video,
            mode=self._mode,
            fps=round(self._fps, 3),
            frames_evaluated=frames_evaluated,
            generated_at=datetime.now(timezone.utc).isoformat(),
            zones=zone_reports,
        )
