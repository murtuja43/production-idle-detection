# Production Idle Detection

Detect industrial production **idle time** from video using dense optical flow,
optionally corroborated by a lightweight **Isolation Forest** anomaly model.

The system measures motion inside configured production zones, flags idle
periods, renders annotated video, logs per-frame results, and produces per-zone
reports. It ships three operating modes (`optical_flow`, `ml`, `combined`) and a
clean, mock-backed architecture for future PLC/ERP integration.

> Status: MVP — Method 1 (visual) fully implemented; Method 2 (PLC + ERP) is a
> documented, tested extension point (interfaces + mocks + reconciler), not a
> live integration.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Detection Modes](#detection-modes)
- [Configuration](#configuration)
- [Optical-Flow Pipeline](#optical-flow-pipeline)
- [Machine-Learning Pipeline](#machine-learning-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Reporting](#reporting)
- [Evaluation](#evaluation)
- [Future PLC / ERP Integration (Method 2)](#future-plc--erp-integration-method-2)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

The project targets two independent methods for productivity / idle-time
detection:

- **Method 1 — Visual detection (implemented).** Dense optical flow measures
  pixel movement between consecutive frames. If movement inside a production zone
  stays below a configurable threshold for a configurable duration, the zone is
  marked idle. Thresholds are configured independently per zone. An Isolation
  Forest can be trained on motion features to corroborate the threshold decision.
- **Method 2 — PLC + ERP integration (future).** The PLC knows whether the
  conveyor is actually running; the ERP holds the planned schedule. The system is
  designed to reconcile **PLC state + ERP schedule + video analysis** (e.g. "PLC
  running but video idle → anomaly"; "ERP planned stop → no alert"). Only clean
  interfaces, mock clients, and the reconciliation logic are provided for now.

### Production Zones

`CMUS`, `COP`, `COK`, `CSK`, `CSLT` — each with an independent ROI, motion
threshold, idle duration, and sensitivity. `CMUS` (welding) adds spark/glare
handling: ROI masking, brightness filtering, optional saturation (colour) gating,
bright connected-component filtering, and a lower default sensitivity.

---

## Architecture

```text
                         ┌──────────────────────────────┐
                         │        configs/*.yaml         │
                         │  (validated AppConfig dataclass)│
                         └───────────────┬───────────────┘
                                         │
        Video ─► VideoProcessor ─► MotionPipeline  (the shared engine)
                 (read/resize/write)   grayscale ─► Farneback dense flow
                                       ─► per-zone motion magnitude
                                          (ROI mask + CMUS spark/glare filter)
                                         │
              ┌──────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
     IdleDetector              FeatureExtractor +            MlIdleClassifier
   (threshold + duration)      FeatureDatasetBuilder         (rolling window ─►
        OPTICAL FLOW            (training dataset CSV)         per-zone IsolationForest)
              │                           │                           │
              └─────────────► ModeEvaluator ◄──────────────────────────┘
                       (optical_flow / ml / combined)
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                        ▼                        ▼
         OverlayRenderer           CsvIdleLogger            ReportAggregator
        (annotated video)        (per-frame CSV)         (CSV + JSON + PNG)

   Method 2 (future, not wired into the loop):
     PlcClient ─┐
                ├─► ProductionReconciler ─► alert verdict per zone
     ErpClient ─┘     (uses the video idle verdict from Method 1)
```

`MotionPipeline` is the single source of motion measurement. Optical-flow
detection, ML training feature extraction, and ML inference all consume it, so
motion is measured identically everywhere (no duplicated optical-flow logic).

---

## Repository Structure

```text
production-idle-detection/
├── configs/
│   └── default.yaml             # validated, documented configuration
├── data/
│   ├── videos/                  # input videos (sample.mp4 included)
│   ├── processed/               # annotated output videos
│   ├── models/                  # trained models + metadata
│   └── ground_truth_example.csv # evaluation ground-truth format example
├── outputs/                     # CSV logs and reports
├── src/
│   ├── detection/               # idle_detector, combined, evaluator
│   ├── erp/                     # ErpClient ABC + Null/Mock (Method 2)
│   ├── features/                # extractor, dataset
│   ├── integration/            # reconciliation (Method 2 blueprint)
│   ├── evaluation/              # metrics, ground truth, plots (standalone)
│   ├── ml/                      # model (persistence), inference
│   ├── optical_flow/            # dense_flow (+ spark/glare filtering)
│   ├── pipeline/                # motion_pipeline (shared engine)
│   ├── plc/                     # PlcClient ABC + Null/Mock (Method 2)
│   ├── preprocessing/           # roi, video_loader
│   ├── reporting/               # report aggregation + output
│   ├── training/                # trainer
│   ├── utils/                   # config, csv_logger, logger
│   └── visualization/           # overlay
├── tests/                       # unit + integration tests
├── main.py                      # inference CLI
├── train.py                     # training CLI
├── evaluate.py                  # evaluation CLI (standalone)
├── requirements.txt
└── README.md
```

---

## Installation

Python 3.11+ (developed and tested on 3.12). Dependencies: OpenCV, NumPy,
scikit-learn, joblib, Matplotlib, PyYAML, tqdm. No deep-learning frameworks.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Optical-flow mode (no model required)
python main.py --config configs/default.yaml --video data/videos/sample.mp4 --mode optical_flow

# ML / combined mode (requires a trained model — see Training)
python main.py --config configs/default.yaml --video data/videos/sample.mp4 --mode ml
python main.py --config configs/default.yaml --video data/videos/sample.mp4 --mode combined
```

CLI flags: `--config`, `--video` (required), `--mode`, `--model`, `--metadata`.
`--mode` overrides `detection.mode`; `--model`/`--metadata` override the configured
model paths.

Outputs (per run):

- Annotated video: `data/processed/<name>_processed.mp4`
- Per-frame CSV: `outputs/idle_detection_log.csv`
- Reports: `outputs/idle_report.csv`, `outputs/idle_report.json`,
  `outputs/idle_report.png`

---

## Detection Modes

| Mode          | Final idle decision                                                |
|---------------|-------------------------------------------------------------------|
| `optical_flow`| Threshold + duration optical-flow detector (Method 1 baseline).   |
| `ml`          | Isolation Forest anomaly over a rolling window of motion features. |
| `combined`    | Fuse optical flow **and** ML via `ml.combine.strategy`.           |

Key guarantees:

- The optical-flow detector **always runs** and is logged in every mode, so the
  Method 1 signal is never lost — ML is strictly additive.
- In `ml`/`combined`, the decision **falls back to optical flow** until a zone's
  rolling window is full, and for any zone without a trained model.
- `combined` strategy `and` (default) requires both signals to agree (high
  precision); `or` fires when either does (high recall).

The per-frame CSV in `ml`/`combined` mode adds transparency columns: `mode`,
`optical_flow_is_idle`, `ml_window_ready`, `ml_is_anomaly`, `ml_score`. In
`optical_flow` mode the CSV schema is exactly the Phase 1 schema.

---

## Configuration

`configs/default.yaml` is fully validated at load time with clear error messages
(ROI bounds, thresholds, durations, feature names, contamination range,
max_samples, spark thresholds, mode/strategy enums). New sections are optional —
older configs without them still load with sensible defaults.

```yaml
detection:
  mode: optical_flow            # optical_flow | ml | combined

features:                       # shared by training AND inference
  window_size: 30               # frames per feature window
  step: 15                      # window stride for dataset generation
  features: [mean, std, max, min, active_ratio, mean_delta]

ml:
  model_path: data/models/isolation_forest.joblib
  metadata_path: data/models/isolation_forest.metadata.json
  combine:
    strategy: and               # and | or

training:
  contamination: auto           # 'auto' or float in (0, 0.5]
  random_state: 42
  n_estimators: 100
  max_samples: auto             # 'auto' | int > 0 | float in (0, 1]
  feature_csv: data/processed/features.csv
  model_output: data/models/isolation_forest.joblib
  metadata_output: data/models/isolation_forest.metadata.json

logging:
  report_csv_filename: idle_report.csv
  report_json_filename: idle_report.json
  report_chart: true
  report_chart_filename: idle_report.png
```

Example zone (CMUS with spark + optional colour gate):

```yaml
CMUS:
  enabled: true
  roi: { x: 60, y: 80, width: 260, height: 220 }
  motion_threshold: 1.15
  idle_duration_seconds: 8.0
  sensitivity: 0.65             # lower => less sensitive to motion
  mask_path: null               # optional grayscale ROI mask
  spark_filter:
    enabled: true
    brightness_threshold: 230   # 0-255
    min_component_area: 4
    dilate_iterations: 1
    kernel_size: 3
    # saturation_threshold: 60  # optional: only suppress low-saturation glare
```

---

## Optical-Flow Pipeline

1. `VideoProcessor` reads frames (0-based index, accurate timestamps), optionally
   resizing, and writes the annotated output.
2. `MotionPipeline` converts to grayscale and computes **Farneback dense optical
   flow** between consecutive frames.
3. For each zone, the per-pixel flow magnitude inside the ROI is averaged, after:
   - applying an optional binary ROI mask (nearest-neighbour resized to the ROI),
   - excluding CMUS spark/glare pixels (bright, and optionally low-saturation),
   - scaling by the zone's `sensitivity`.
4. `IdleDetector` marks a zone idle once the motion score stays below
   `motion_threshold` for `idle_duration_seconds`.

### CMUS spark / glare handling

Welding produces bright transient pixels that look like motion. The spark filter
builds a binary mask of bright pixels (`>= brightness_threshold`), optionally
restricted to low-saturation pixels (`<= saturation_threshold`) so that genuinely
coloured moving parts are preserved, then dilates and keeps connected components
of at least `min_component_area`. Masked pixels are excluded from motion scoring.

---

## Machine-Learning Pipeline

- **Features** (`src/features/extractor.py`): per-zone motion magnitudes are
  aggregated over a sliding window. Available: `mean`, `std`, `max`, `min`,
  `median`, `range`, `energy`, `active_ratio` (fraction of frames at/above the
  zone threshold), `mean_delta` (mean absolute frame-to-frame change). The
  selected set is shared by training and inference.
- **Dataset** (`src/features/dataset.py`): `FeatureDatasetBuilder` drives
  `MotionPipeline` over one or more videos and writes a feature CSV
  (`source, zone, window_index, start/end_frame, start/end_time` + feature
  columns).
- **Model** (`src/ml/model.py`): one Isolation Forest **per zone** (zones differ
  too much to share a model), bundled into a single joblib file, with a JSON
  metadata sidecar (feature names, window/step, hyperparameters, per-zone sample
  counts, scikit-learn version, timestamp, source). Metadata is the source of
  truth for the feature layout at inference.
- **Inference** (`src/ml/inference.py`): `MlIdleClassifier` keeps a per-zone
  rolling window and, once full, builds the feature vector with the same
  extractor and queries the model.

---

## Training

```bash
# Generate features AND train in one command
python train.py --config configs/default.yaml --videos data/videos/sample.mp4

# Only generate the feature CSV
python train.py --config configs/default.yaml --videos a.mp4 b.mp4 --extract-only

# Train from an existing feature CSV
python train.py --config configs/default.yaml --skip-extract \
    --features-csv data/processed/features.csv
```

Configurable via the `features`/`training` config sections or CLI overrides:
`--contamination`, `--random-state`, `--n-estimators`, `--window-size`, `--step`,
`--features`, `--features-csv`, `--model-output`, `--metadata-output`.

Produces: the feature CSV, the trained model (`.joblib`), and the metadata
(`.json`).

> The default `features.window_size: 30` suits real footage. For the short
> bundled `sample.mp4` (60 frames), train with `--window-size 10 --step 5`.

---

## Inference

`main.py` builds the `MotionPipeline`, runs the `ModeEvaluator` for the selected
mode, writes the annotated video and per-frame CSV, and emits reports. ML and
combined modes require a trained model (a clear error is raised if it is
missing).

---

## Reporting

Every run produces a per-zone report in CSV and JSON, plus a stacked bar chart
(PNG) of idle vs active time. Reported metrics per zone: frames evaluated, total
time, **idle time**, active time, **idle event count**, **average motion**, and
**anomaly count**; the JSON also includes a top-level summary (mode, video, fps,
totals). Charting failures never abort a run.

---

## Evaluation

`evaluate.py` is a **standalone** utility (it does not import or modify the
detection pipeline) that scores predictions against labelled ground truth and
visualizes the comparison.

```bash
python evaluate.py \
    --predictions outputs/idle_detection_log.csv \
    --ground-truth data/ground_truth_example.csv \
    --default-label active
```

It reads the per-frame detection CSV (its `is_idle` column is the prediction for
whichever mode produced it) and a ground-truth CSV of labelled timestamp ranges,
aligns them per zone and timestamp, and reports — per zone and overall —
**Accuracy, Precision, Recall, F1, and the confusion matrix** (positive class =
idle). Outputs:

- `outputs/evaluation_metrics.json` — summary + per-zone + overall metrics
- `outputs/evaluation_metrics.csv` — per-zone (and overall) metrics table
- `outputs/evaluation_comparison.png` — predicted vs actual idle timeline per zone

Ground-truth CSV format (columns are case-insensitive; a row without a `zone`
applies to all zones):

```csv
zone,start_seconds,end_seconds,label
CMUS,2.0,4.0,idle
COP,3.0,6.0,idle
COK,0.0,6.0,active
```

Accepted aliases: start (`start`/`start_seconds`/`start_time`), end
(`end`/`end_seconds`/`end_time`), label (`label`/`state`/`is_idle`; values
idle/active, 1/0, true/false). Frames outside every interval take
`--default-label` (`active` default, or `idle`, or `skip` to exclude them from
scoring).

## Future PLC / ERP Integration (Method 2)

Method 2 is intentionally **not wired into the main pipeline**. It is fully
specified as contracts + mocks + reconciliation logic so real clients can be
dropped in later without touching detection code:

- `src/plc/interface.py` — `PlcClient` (ABC), `NullPlcClient`, `MockPlcClient`.
- `src/erp/interface.py` — `ErpClient` (ABC), `NullErpClient`, `MockErpClient`.
- `src/integration/reconciliation.py` — `ProductionReconciler` combines the
  video idle verdict (Method 1) with PLC state and the ERP schedule.

Decision matrix (per zone), given the video idle verdict:

| ERP order | PLC running | Video idle | Result               | Alert |
|-----------|-------------|------------|----------------------|-------|
| none      | any         | any        | `PLANNED_STOP`       | no    |
| present   | running     | active     | `RUNNING`            | no    |
| present   | running     | idle       | `IDLE_WHILE_RUNNING` | yes   |
| present   | stopped     | idle       | `UNPLANNED_STOP`     | yes   |
| present   | stopped     | active     | `SENSOR_DISAGREEMENT`| yes   |
| present   | unknown     | idle       | `IDLE_NO_PLC`        | yes   |
| present   | unknown     | active     | `RUNNING`            | no    |

To implement a real integration: subclass `PlcClient` / `ErpClient` with real
transports (e.g. Modbus/OPC-UA, ERP/MES REST), then feed the per-zone idle
verdict into `ProductionReconciler` and route alerts to your notification system.

---

## Testing

```bash
python -m unittest discover -s tests
```

Coverage: config parsing + validation (incl. backward compatibility), the shared
`MotionPipeline`, optical-flow idle-detector state transitions, CMUS spark/colour
gating, feature extraction and dataset generation, model persistence and
training, ML inference warmup, combine logic and the mode evaluator (incl.
optical-flow backward-compatibility), reporting, PLC/ERP reconciliation, overlay
rendering, evaluation (metrics, ground-truth parsing, alignment, plotting), and
end-to-end optical-flow / ml / combined smoke tests.

---

## Limitations

- **Anomaly framing.** The Isolation Forest is unsupervised; it flags *unusual*
  motion windows. It is most effective when trained on footage where idle periods
  are the minority. It does not replace the optical-flow detector — `combined`
  mode is recommended for production use.
- **Camera assumptions.** Dense optical flow assumes a fixed camera and roughly
  stable lighting. Camera shake, exposure swings, or PTZ movement degrade
  accuracy; ROIs are in (optionally resized) frame pixel coordinates.
- **Per-video, single-process.** Processing is CPU-bound and runs one video at a
  time; there is no streaming/RTSP ingestion or GPU acceleration.
- **Frame-count metadata.** Some containers misreport frame counts (progress bar
  only); timestamps are derived from FPS.
- **Method 2 is not live.** PLC/ERP are mock-backed; no real industrial transport
  is implemented.
- **Reports are time-approximate.** Idle/active seconds are derived from frame
  counts × frame period (1/fps).

---

## Future Work

- Real PLC (Modbus/OPC-UA) and ERP/MES clients behind the existing interfaces.
- Streaming/RTSP ingestion and multi-camera orchestration.
- Alerting/notification sink for reconciliation verdicts.
- Optional supervised labelling workflow to complement unsupervised anomalies.
- Per-zone model evaluation metrics and threshold auto-tuning.
```
