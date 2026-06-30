"""Ground-truth loading and timestamp label lookup.

The ground-truth CSV holds labelled timestamp ranges. Recognised columns
(case-insensitive):

- zone        (optional; a row without a zone applies to all zones)
- start       | start_seconds | start_time
- end         | end_seconds   | end_time
- label       | state         | is_idle   (idle/active, 1/0, true/false)
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_START_COLUMNS = ("start", "start_seconds", "start_time")
_END_COLUMNS = ("end", "end_seconds", "end_time")
_LABEL_COLUMNS = ("label", "state", "is_idle")
_IDLE_TOKENS = {"idle", "true", "1", "yes", "stopped", "stop"}
_ACTIVE_TOKENS = {"active", "false", "0", "no", "running", "run"}


def parse_label(value: object) -> bool:
    """Parse a label cell into ``is_idle`` (True = idle)."""
    text = str(value).strip().lower()
    if text in _IDLE_TOKENS:
        return True
    if text in _ACTIVE_TOKENS:
        return False
    try:
        return float(text) != 0.0
    except ValueError as error:
        raise ValueError(
            f"Unrecognized ground-truth label: {value!r}. "
            f"Use idle/active, 1/0, or true/false."
        ) from error


@dataclass(frozen=True)
class GroundTruthInterval:
    """A labelled time range, optionally scoped to a single zone."""

    zone: str | None
    start: float
    end: float
    is_idle: bool

    def matches(self, zone: str, timestamp: float) -> bool:
        """Whether this interval applies to ``zone`` and contains ``timestamp``."""
        if self.zone is not None and self.zone != zone:
            return False
        return self.start <= timestamp < self.end


def _resolve_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_ground_truth(path: str | Path) -> list[GroundTruthInterval]:
    """Load labelled intervals from a ground-truth CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV does not exist: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Ground-truth CSV is empty: {csv_path}")

        start_col = _resolve_column(reader.fieldnames, _START_COLUMNS)
        end_col = _resolve_column(reader.fieldnames, _END_COLUMNS)
        label_col = _resolve_column(reader.fieldnames, _LABEL_COLUMNS)
        zone_col = _resolve_column(reader.fieldnames, ("zone",))
        if start_col is None or end_col is None or label_col is None:
            raise ValueError(
                "Ground-truth CSV must have start, end, and label columns. "
                f"Found: {reader.fieldnames}"
            )

        intervals: list[GroundTruthInterval] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                start = float(row[start_col])
                end = float(row[end_col])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid start/end on ground-truth row {line_number}."
                ) from error
            if end < start:
                raise ValueError(
                    f"Ground-truth row {line_number}: end ({end}) < start ({start})."
                )
            zone_value = row.get(zone_col) if zone_col else None
            zone = zone_value.strip() if zone_value and zone_value.strip() else None
            intervals.append(
                GroundTruthInterval(
                    zone=zone,
                    start=start,
                    end=end,
                    is_idle=parse_label(row[label_col]),
                )
            )

    if not intervals:
        raise ValueError(f"Ground-truth CSV has no rows: {csv_path}")
    return intervals


def label_at(
    intervals: Iterable[GroundTruthInterval],
    zone: str,
    timestamp: float,
    default: bool | None,
) -> bool | None:
    """Return the ground-truth ``is_idle`` for a zone at a timestamp.

    If several intervals overlap the timestamp, idle takes precedence. When no
    interval matches, ``default`` is returned (use ``None`` to mark the frame as
    unlabelled so it is excluded from scoring).
    """
    matched = [iv for iv in intervals if iv.matches(zone, timestamp)]
    if not matched:
        return default
    return any(iv.is_idle for iv in matched)
