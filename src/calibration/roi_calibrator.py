"""Pure (GUI-free) helpers for ROI calibration.

The interactive window lives in ``calibrate_rois.py``; everything testable lives
here: rectangle normalization/clamping, ROI validation, and a *surgical* YAML
updater that rewrites only the ``zones.<zone>.roi`` x/y/width/height values while
leaving every other byte of the config (comments, ordering, formatting) intact.
This module never imports or touches the detection pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import REQUIRED_ZONES

# Calibration order requested for the five production zones.
ZONE_ORDER: tuple[str, ...] = REQUIRED_ZONES

_ROI_KEY_INDEX = {"x": 0, "y": 1, "width": 2, "height": 3}


def read_first_frame(video_path: str | Path) -> np.ndarray:
    """Return the first frame (BGR) of a video, raising on failure."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(f"Could not read the first frame of: {path}")
        return frame
    finally:
        capture.release()


def normalize_rect(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """Normalize two drag corners into ``(x, y, width, height)`` with w,h >= 0."""
    x = int(min(x0, x1))
    y = int(min(y0, y1))
    width = int(abs(x1 - x0))
    height = int(abs(y1 - y0))
    return x, y, width, height


def clamp_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Clamp an ROI so it stays fully inside ``frame_width`` x ``frame_height``."""
    x = max(0, min(int(x), frame_width - 1))
    y = max(0, min(int(y), frame_height - 1))
    width = max(0, min(int(width), frame_width - x))
    height = max(0, min(int(height), frame_height - y))
    return x, y, width, height


def validate_roi(
    x: int,
    y: int,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
    zone: str = "ROI",
) -> None:
    """Validate an ROI: integer, positive size, inside image bounds."""
    values = {"x": x, "y": y, "width": width, "height": height}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{zone}: {name} must be an integer (got {value!r}).")
    if width <= 0 or height <= 0:
        raise ValueError(f"{zone}: width and height must be positive.")
    if x < 0 or y < 0:
        raise ValueError(f"{zone}: x and y must be >= 0.")
    if x + width > frame_width or y + height > frame_height:
        raise ValueError(
            f"{zone}: ROI ({x}, {y}, {width}, {height}) exceeds frame bounds "
            f"{frame_width}x{frame_height}."
        )


def update_roi_values(
    text: str,
    zone_rois: dict[str, tuple[int, int, int, int]],
) -> str:
    """Return ``text`` with each zone's roi x/y/width/height replaced.

    Only the four numeric value lines per zone change; comments, ordering, and
    all other content are preserved exactly. Raises if a requested zone or any of
    its roi fields cannot be located.
    """
    lines = text.split("\n")

    zones_index = next(
        (i for i, line in enumerate(lines) if re.match(r"^zones:\s*(#.*)?$", line)),
        None,
    )
    if zones_index is None:
        raise ValueError("Could not find a top-level 'zones:' section in the config.")

    zone_indent = _first_child_indent(lines, zones_index)
    if zone_indent is None:
        raise ValueError("The 'zones:' section is empty.")

    replaced: dict[str, set[str]] = {zone: set() for zone in zone_rois}
    current_zone: str | None = None
    roi_indent: int | None = None

    index = zones_index + 1
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue

        indent = _indent_of(line)
        if indent == 0:
            break  # a new top-level key ends the zones section
        if indent == zone_indent:
            match = re.match(r"^\s*([A-Za-z0-9_]+):\s*(#.*)?$", line)
            current_zone = match.group(1) if match else None
            roi_indent = None
            index += 1
            continue

        if current_zone in zone_rois:
            if roi_indent is None:
                if re.match(r"^\s*roi:\s*(#.*)?$", line):
                    roi_indent = indent
                index += 1
                continue
            if indent <= roi_indent:
                roi_indent = None
                continue  # re-evaluate this line as a sibling/zone header
            match = re.match(
                r"^(\s+)(x|y|width|height):\s*(\S+)(\s*#.*)?\s*$", line
            )
            if match:
                indent_str, key, _old_value, comment = match.groups()
                value = zone_rois[current_zone][_ROI_KEY_INDEX[key]]
                lines[index] = f"{indent_str}{key}: {value}{comment or ''}"
                replaced[current_zone].add(key)
        index += 1

    _ensure_all_replaced(replaced)
    return "\n".join(lines)


def apply_calibration_to_config(
    config_path: str | Path,
    zone_rois: dict[str, tuple[int, int, int, int]],
    frame_size: tuple[int, int],
) -> None:
    """Validate ROIs against the frame and write them into ``config_path``."""
    frame_width, frame_height = frame_size
    for zone, (x, y, width, height) in zone_rois.items():
        validate_roi(x, y, width, height, frame_width, frame_height, zone)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    original = path.read_text(encoding="utf-8")
    path.write_text(update_roi_values(original, zone_rois), encoding="utf-8")


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _first_child_indent(lines: list[str], parent_index: int) -> int | None:
    for line in lines[parent_index + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = _indent_of(line)
        return indent if indent > 0 else None
    return None


def _ensure_all_replaced(replaced: dict[str, set[str]]) -> None:
    expected = set(_ROI_KEY_INDEX)
    missing = {
        zone: sorted(expected - keys)
        for zone, keys in replaced.items()
        if keys != expected
    }
    if missing:
        details = "; ".join(f"{zone}: missing {keys}" for zone, keys in missing.items())
        raise ValueError(
            f"Could not update all ROI fields in the config ({details}). "
            "Check that each requested zone has an roi block with x/y/width/height."
        )
