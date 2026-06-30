# Production Idle Detection

Production-quality MVP for detecting industrial production idle time from video.

The system measures motion inside configured production zones using dense
optical flow and flags idle periods. Phase 2 adds a machine-learning pipeline
(per-zone Isolation Forests) on top of the optical-flow engine, offering three
operating modes: `optical_flow`, `ml`, and `combined`.

## Methods

This project targets two independent detection methods:

- **Method 1 — Visual detection (implemented).** Dense optical flow measures
  pixel movement between consecutive frames. If movement inside a production
  zone stays below a configurable threshold for a configurable duration, the
  zone is marked idle. Thresholds are configured independently per zone, and an
  Isolation Forest anomaly model can corroborate the threshold decision.
- **Method 2 — PLC + ERP integration (future).** The PLC knows whether the
  conveyor is actually running; the ERP holds the planned schedule. A future
  system will reconcile PLC state + ERP schedule + video analysis (e.g. "PLC
  running but video idle → anomaly"; "PLC planned stop → no alert"). Only clean
  interfaces are provided for now under `src/plc/` and `src/erp/`.

## Supported Production Zones

- `CMUS`
- `COP`
- `COK`
- `CSK`
- `CSLT`

Each zone is configured independently with an ROI rectangle, motion threshold,
idle duration, and sensitivity.

`CMUS` (welding area) supports extra spark/glare handling:

- Optional ROI mask image
- Brightness threshold filtering
- Bright connected-component filtering
- Lower default sensitivity

## Architecture

```text
Video
  -> VideoProcessor (read/resize/write)
  -> MotionPipeline  ── the shared engine ──────────────┐
       (grayscale -> Farneback dense flow -> per-zone     │
        motion magnitude, with ROI mask + spark filter)   │
                                                           │
  ├─ Optical-flow path: IdleDetector (threshold+duration) │
  └─ ML path: FeatureExtractor -> MlIdleClassifier        │
                                  (rolling window -> IF)   │
  -> ModeEvaluator (optical_flow / ml / combined)  ◄───────┘
  -> OverlayRenderer + CsvIdleLogger
```

`MotionPipeline` is the single source of motion measurement. Optical-flow
detection, ML feature extraction (training), and ML inference all consume it, so
motion is measured identically everywhere.

## Installation

Use Python 3.11+ (developed/tested on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Detection Modes

Set `detection.mode` in the config (or override with `--mode`):

| Mode          | Final idle decision                                              |
|---------------|-----------------------------------------------------------------|
| `optical_flow`| Threshold + duration optical-flow detector (Phase 1 behavior).  |
| `ml`          | Isolation Forest anomaly per rolling window of motion features. |
| `combined`    | Fuse optical flow **and** ML via `ml.combine.strategy`.         |

The optical-flow detector **always runs**, so its signal is preserved and logged
in every mode. In `ml`/`combined` modes the decision falls back to optical flow
until each zone's rolling window is full (and for any zone without a trained
model). `combined` strategy `and` (default) requires both signals to agree
(high precision); `or` fires when either does (high recall).

## Usage (inference)

Optical-flow mode (no model required):

```bash
python main.py --config configs/default.yaml --video data/videos/sample.mp4
```

ML or combined mode (requires a trained model):

```bash
python main.py --config configs/default.yaml --video data/videos/sample.mp4 --mode combined
python main.py --config configs/default.yaml --video data/videos/sample.mp4 --mode ml \
    --model data/models/isolation_forest.joblib \
    --metadata data/models/isolation_forest.metadata.json
```

Outputs:

- Processed video: `data/processed/<name>_processed.mp4`
- CSV log: `outputs/idle_detection_log.csv`

In `ml`/`combined` mode the CSV gains transparency columns: `mode`,
`optical_flow_is_idle`, `ml_window_ready`, `ml_is_anomaly`, `ml_score`. In
`optical_flow` mode the CSV schema is exactly the Phase 1 schema.

## Machine Learning Pipeline

### Feature extraction

For each zone, the per-frame motion magnitude stream is aggregated over a sliding
window (`features.window_size` frames). Available features
(`src/features/extractor.py`): `mean`, `std`, `max`, `min`, `median`, `range`,
`energy`, `active_ratio` (fraction of frames at/above the zone threshold), and
`mean_delta` (mean absolute frame-to-frame change). The selected set is shared by
training and inference, guaranteeing identical feature computation.

### Dataset generation

`FeatureDatasetBuilder` drives `MotionPipeline` over one or more videos, slides
windows (`features.step` stride), and writes a feature CSV with metadata columns
(`source, zone, window_index, start_frame, end_frame, start_time, end_time`) plus
one column per feature.

### Model & metadata

Training fits **one Isolation Forest per zone** (zones differ too much to share a
model). The estimators are bundled into a single joblib file; a JSON metadata
sidecar records feature names, window/step, hyperparameters, per-zone sample
counts, scikit-learn version, timestamp, and source. At inference the metadata is
the source of truth for the feature layout.

## Training workflow

```bash
# 1) Generate features AND train in one command
python train.py --config configs/default.yaml --videos data/videos/sample.mp4

# Only generate the feature CSV
python train.py --config configs/default.yaml --videos a.mp4 b.mp4 --extract-only

# Train from an existing feature CSV
python train.py --config configs/default.yaml --skip-extract \
    --features-csv data/processed/features.csv
```

Configurable training parameters (config `training`/`features` sections, or CLI
overrides): `--contamination`, `--random-state`, `--n-estimators`,
`--window-size`, `--step`, `--features`, `--features-csv`, `--model-output`,
`--metadata-output`.

Produces:

- Feature CSV: `data/processed/features.csv`
- Trained model: `data/models/isolation_forest.joblib`
- Metadata: `data/models/isolation_forest.metadata.json`

## Configuration

The default config is `configs/default.yaml`. Phase 1 zone example:

```yaml
CMUS:
  enabled: true
  roi: { x: 60, y: 80, width: 260, height: 220 }
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

Phase 2 sections:

```yaml
detection:
  mode: optical_flow        # optical_flow | ml | combined

features:
  window_size: 30
  step: 15
  features: [mean, std, max, min, active_ratio, mean_delta]

ml:
  model_path: data/models/isolation_forest.joblib
  metadata_path: data/models/isolation_forest.metadata.json
  combine:
    strategy: and           # and | or

training:
  contamination: auto
  random_state: 42
  n_estimators: 100
  max_samples: auto
  feature_csv: data/processed/features.csv
  model_output: data/models/isolation_forest.joblib
  metadata_output: data/models/isolation_forest.metadata.json
```

These sections are optional: older Phase 1 configs without them still load with
sensible defaults.

### Motion Threshold, Sensitivity, ROI Masks

The detector computes dense optical-flow magnitude inside each ROI; if the
filtered, sensitivity-scaled motion score stays below `motion_threshold` for at
least `idle_duration_seconds`, the zone is idle. Sensitivity scales the measured
score (`effective_motion = raw_motion * sensitivity`); lower values reduce
sensitivity to motion (useful for CMUS sparks). Set `mask_path` to a grayscale
image (non-zero pixels included, zero pixels ignored).

## Project Structure

```text
production-idle-detection/
├── configs/default.yaml
├── data/{videos,processed,models}/
├── outputs/
├── src/
│   ├── detection/          # idle_detector, combined, evaluator
│   ├── erp/                # ERP interface (Method 2, future)
│   ├── features/           # extractor, dataset
│   ├── ml/                 # model (persistence), inference
│   ├── optical_flow/       # dense_flow
│   ├── pipeline/           # motion_pipeline (shared engine)
│   ├── plc/                # PLC interface (Method 2, future)
│   ├── preprocessing/      # roi, video_loader
│   ├── training/           # trainer
│   ├── utils/              # config, csv_logger, logger
│   └── visualization/      # overlay
├── tests/
├── main.py                 # inference CLI
├── train.py                # training CLI
├── README.md
└── requirements.txt
```

## Testing

```bash
python -m unittest discover -s tests
```

Coverage includes: config parsing (Phase 1 + Phase 2, backward compatibility),
idle-detector state transitions, the shared `MotionPipeline`, feature extraction
and dataset generation, model persistence and training, ML inference warmup,
combine logic, the mode evaluator (incl. optical-flow backward compatibility),
and end-to-end optical-flow / ml / combined smoke tests.

## Scope

Implemented (Phase 1):

- Farneback dense optical flow, multi-zone ROI handling, per-zone idle detection
- CMUS brightness-based spark filtering, optional ROI masks
- Overlay rendering, processed video output, CSV logging
- PLC/ERP interfaces for Method 2

Implemented (Phase 2):

- Motion feature extraction and sliding-window aggregation
- Dataset generation from video, feature CSV
- Per-zone Isolation Forest training with configurable parameters
- Model persistence (joblib) + metadata (JSON), model loading
- ML inference and `optical_flow` / `ml` / `combined` modes
- Training CLI and comprehensive unit tests

Not implemented (future):

- Real PLC integration
- Real ERP integration
```
