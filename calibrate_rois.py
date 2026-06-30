"""Interactive ROI calibration for new factory cameras.

Draw the five production-zone ROIs on the first frame of a video, then write them
into the config so the detector adapts to a new camera with no code changes.

    python calibrate_rois.py --video data/videos/conveyor_demo.mp4

Per zone (in order CMUS, COP, COK, CSK, CSLT):
    - drag a rectangle with the mouse
    - ENTER / SPACE  confirm
    - R              redraw the current ROI
    - ESC            cancel calibration

This utility only edits ``zones.<zone>.roi`` in the config; it does not import or
modify any detection logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src.calibration.roi_calibrator import (
    ZONE_ORDER,
    apply_calibration_to_config,
    clamp_rect,
    normalize_rect,
    read_first_frame,
    validate_roi,
)

_WINDOW = "ROI Calibration"
_ENTER_KEYS = (13, 10)
_SPACE_KEY = 32
_ESC_KEY = 27
_CONFIRMED_COLOR = (0, 200, 0)
_CURRENT_COLOR = (0, 215, 255)


def parse_args() -> argparse.Namespace:
    """Parse calibration CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Interactively calibrate production-zone ROIs from a video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config file to update (only zones.<zone>.roi is changed).",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1280,
        help="Downscale the preview window to at most this width (coordinates are "
        "mapped back to the native resolution).",
    )
    return parser.parse_args()


def _draw_overlay(
    image,
    zone_name: str,
    confirmed_display: dict[str, tuple[int, int, int, int]],
    current: tuple[int, int, int, int] | None,
):
    canvas = image.copy()
    for name, (x, y, w, h) in confirmed_display.items():
        cv2.rectangle(canvas, (x, y), (x + w, y + h), _CONFIRMED_COLOR, 2)
        cv2.putText(
            canvas, name, (x + 3, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _CONFIRMED_COLOR, 2, cv2.LINE_AA,
        )
    if current is not None and current[2] > 0 and current[3] > 0:
        x, y, w, h = current
        cv2.rectangle(canvas, (x, y), (x + w, y + h), _CURRENT_COLOR, 2)

    lines = [
        f"Draw ROI for: {zone_name}",
        "Drag mouse | ENTER/SPACE confirm | R redraw | ESC cancel",
    ]
    for row, text in enumerate(lines):
        y = 24 + row * 26
        cv2.rectangle(canvas, (0, y - 20), (12 + 9 * len(text), y + 8), (0, 0, 0), -1)
        cv2.putText(
            canvas, text, (6, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return canvas


def _select_zone_roi(
    display_frame,
    zone_name: str,
    confirmed_display: dict[str, tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Run the interactive loop for one zone. Returns a display-space rect or None."""
    state: dict[str, object] = {"start": None, "rect": None, "drawing": False}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(start=(x, y), rect=(x, y, 0, 0), drawing=True)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            x0, y0 = state["start"]  # type: ignore[misc]
            state["rect"] = normalize_rect(x0, y0, x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            x0, y0 = state["start"]  # type: ignore[misc]
            state["rect"] = normalize_rect(x0, y0, x, y)
            state["drawing"] = False

    cv2.setMouseCallback(_WINDOW, on_mouse)
    while True:
        canvas = _draw_overlay(
            display_frame, zone_name, confirmed_display, state["rect"]  # type: ignore[arg-type]
        )
        cv2.imshow(_WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        rect = state["rect"]
        if key in _ENTER_KEYS or key == _SPACE_KEY:
            if rect is not None and rect[2] > 0 and rect[3] > 0:  # type: ignore[index]
                return rect  # type: ignore[return-value]
        elif key in (ord("r"), ord("R")):
            state.update(start=None, rect=None, drawing=False)
        elif key == _ESC_KEY:
            return None


def _to_native(
    rect: tuple[int, int, int, int],
    scale: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    native = (
        round(x / scale), round(y / scale), round(w / scale), round(h / scale)
    )
    return clamp_rect(*native, frame_width, frame_height)


def _to_display(rect: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    return round(x * scale), round(y * scale), round(w * scale), round(h * scale)


def calibrate(video_path: str, config_path: str, max_display_width: int) -> int:
    """Run the full calibration workflow. Returns a process exit code."""
    frame = read_first_frame(video_path)
    frame_height, frame_width = frame.shape[:2]
    scale = min(1.0, max_display_width / frame_width)
    display_frame = (
        frame if scale == 1.0
        else cv2.resize(
            frame,
            (round(frame_width * scale), round(frame_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    )

    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
    native_rois: dict[str, tuple[int, int, int, int]] = {}
    display_rois: dict[str, tuple[int, int, int, int]] = {}
    try:
        for zone in ZONE_ORDER:
            selected = _select_zone_roi(display_frame, zone, display_rois)
            if selected is None:
                print("Calibration cancelled. No changes were written.")
                return 1
            native = _to_native(selected, scale, frame_width, frame_height)
            try:
                validate_roi(*native, frame_width, frame_height, zone)
            except ValueError as error:
                print(f"Invalid ROI for {zone}: {error}. Redraw it.")
                # Re-run this zone by retrying the loop iteration.
                selected = _select_zone_roi(display_frame, zone, display_rois)
                if selected is None:
                    print("Calibration cancelled. No changes were written.")
                    return 1
                native = _to_native(selected, scale, frame_width, frame_height)
                validate_roi(*native, frame_width, frame_height, zone)
            native_rois[zone] = native
            display_rois[zone] = _to_display(native, scale)

        # Show the completed layout.
        layout = _draw_overlay(display_frame, "ALL ZONES SET", display_rois, None)
        cv2.imshow(_WINDOW, layout)
        cv2.waitKey(400)
    finally:
        cv2.destroyAllWindows()
        # Flush GUI events so the window closes cleanly on every platform.
        for _ in range(4):
            cv2.waitKey(1)

    print("\nProposed ROIs (native resolution {}x{}):".format(frame_width, frame_height))
    _print_summary(native_rois)

    answer = input("\nSave these ROIs? (Y/N): ").strip().lower()
    if answer not in ("y", "yes"):
        print("Not saved.")
        return 1

    apply_calibration_to_config(config_path, native_rois, (frame_width, frame_height))
    print(f"\nSaved ROIs to {config_path}")
    _print_summary(native_rois)
    return 0


def _print_summary(zone_rois: dict[str, tuple[int, int, int, int]]) -> None:
    for zone in ZONE_ORDER:
        if zone in zone_rois:
            x, y, w, h = zone_rois[zone]
            print(f"  {zone}: ({x}, {y}, {w}, {h})")


def main() -> None:
    """Run the calibration CLI."""
    args = parse_args()
    exit_code = calibrate(args.video, args.config, args.max_display_width)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
