# Production Idle Detection

Production-quality MVP for detecting industrial production idle time from video.

Phase 1 implements visual idle detection using dense optical flow. The system
loads industrial videos, evaluates configured production zones, detects idle
periods, renders overlays, saves processed videos, and exports per-zone CSV logs.

## Supported Production Zones

- `CMUS`
- `COP`
- `COK`
- `CSK`
- `CSLT`

Each zone is configured independently with:

- ROI rectangle
- Motion threshold
- Idle duration
- Sensitivity

`CMUS` supports additional welding-spark handling:

- Optional ROI mask image
- Brightness threshold filtering
- Bright connected-component filtering
- Lower default sensitivity

## Architecture

```text
Video
  -> Preprocessing
  -> ROI Selection
  -> Farneback Dense Optical Flow
  -> Motion Measurement
  -> Idle Detection
  -> Overlay Visualization
  -> CSV Logging
```

PLC and ERP integrations are intentionally not implemented in Phase 1. Clean
interfaces are provided under `src/plc/` and `src/erp/` so real clients can be
plugged in later.

## Installation

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Place an input video under `data/videos/`, for example:

```text
data/videos/sample.mp4
```

Run the detector:

```bash
python main.py --config configs/default.yaml --video data/videos/sample.mp4
```

Outputs:

- Processed video: `data/processed/sample_processed.mp4`
- CSV log: `outputs/idle_detection_log.csv`

## Configuration

The default config is `configs/default.yaml`.

Example zone configuration:

```yaml
CMUS:
  enabled: true
  roi:
    x: 60
    y: 80
    width: 260
    height: 220
  motion_threshold: 1.15
  idle_duration_seconds: 8.0
  sensitivity: 0.65
  mask_path: null
  spark_filter:
    enabled: true
    brightness_threshold: 230
    min_component_area: 4
    dilate_iterations: 1
```

### Motion Threshold

The detector computes dense optical-flow magnitude inside each ROI. If the
filtered motion score remains below `motion_threshold` for at least
`idle_duration_seconds`, the zone is marked idle.

### Sensitivity

Sensitivity scales the measured motion score:

```text
effective_motion = raw_motion * sensitivity
```

Lower values make a detector less sensitive to motion. This is useful for CMUS,
where welding sparks can otherwise cause false activity.

### ROI Masks

Set `mask_path` to a grayscale image with the same aspect as the ROI. Non-zero
pixels are included in motion scoring; zero pixels are ignored.

## Project Structure

```text
production-idle-detection/
├── configs/
│   └── default.yaml
├── data/
│   ├── models/
│   ├── processed/
│   └── videos/
├── outputs/
├── src/
│   ├── detection/
│   ├── erp/
│   ├── features/
│   ├── optical_flow/
│   ├── plc/
│   ├── preprocessing/
│   ├── training/
│   ├── utils/
│   └── visualization/
├── tests/
├── main.py
├── README.md
└── requirements.txt
```

## Testing

Run Phase 1 unit tests:

```bash
python -m unittest discover -s tests
```

The tests cover:

- Config parsing and required zone validation
- Idle detector state transitions
- End-to-end synthetic video processing smoke test

## Phase 1 Scope

Implemented:

- OpenCV Farneback Dense Optical Flow
- Multi-zone ROI handling
- Per-zone idle detection
- CMUS brightness-based spark filtering
- Optional ROI mask support
- Overlay rendering
- Processed video output
- CSV logging
- PLC/ERP interfaces for later phases

Not implemented in Phase 1:

- ML feature training
- Isolation Forest inference
- Real PLC integration
- Real ERP integration

