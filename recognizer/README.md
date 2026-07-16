# Recognizer Runtime

Recognizer Runtime is the runtime path for the 16x16 pressure cushion project. Data collection and label validation stay frozen outside this package; this package only reads frozen model artifacts and implements recognition.

Current default model: `v2_2_candidate`, stored in `recognizer/models/default_model.json`.

V1 and V2 candidate remain available as historical rollback versions.

## Runtime Flow

```text
Pressure Sensor
  -> Baseline Calibration
  -> Occupancy Detection
  -> Human / Object Verification
  -> Seat State Detection
  -> Stable Window Buffer
  -> FeatureExtractor
  -> RF Posture Recognizer
  -> Smoothing
  -> Posture + Confidence + Boundary + Duration
  -> GUI / realtime client
```

## Modules

- `recognizer_api.py`: stable hardware-facing API. External callers use `Recognizer().predict(frame)` and do not load models or call internal modules directly.
- `feature_extractor.py`: shared 264-dim feature interface for training, prediction, and realtime.
- `occupancy_detector.py`: rule-based EMPTY / LOAD_BELOW_THRESHOLD / OBJECT / HUMAN / UNKNOWN gate with explainable pressure features.
- `seat_analyzer.py`: high-level `analyze_seat` gate; posture recognition is only called for stable HUMAN windows.
- `frame_reader.py`: `FrameReader`, `CSVReplayReader`, and configurable `SerialFrameReader` skeleton.
- `seat_detector.py`: empty, sitting down, stabilizing, stable, and standing up state detection.
- `prototype_bank.py`: serializable prototype bank with class thresholds and mirror-aware prototypes.
- `recognizer.py`: `predict_posture(window)` public recognizer interface for prototype banks and probability models such as Random Forest.
- `pipeline.py`: realtime seat detection + rolling window + recognizer pipeline.
- `training.py`: grouped CSV validation and backend comparison.
- `model_artifact.py`: trains and freezes the RF V1 candidate with metadata and save/load consistency checks.
- `rf_recognizer.py`: hybrid RF-primary recognizer with prototype diagnosis.
- `smoothing.py`: realtime majority voting, switch confirmation, and low-confidence boundary handling.
- `realtime_cli.py`: CSV stream replay through seat detection, stable windowing, RF recognition, and smoothing.
- `predict.py`: CLI for replaying a CSV against a saved prototype bank.
- `gui.py`: simple Tkinter CSV playback GUI.

## V1 Label Policy

Recognizer V1 uses the current stable taxonomy:

- Class 5 + merged Class 8 -> `后靠/瘫坐类`
- Class 10 + Class 11 -> `躺卧类`
- Class 11 uses mirror-aware matching internally, but still outputs only `躺卧类`.
- Class 5 and Class 11 can use multiple internal prototypes while keeping one public label.

## Commands

Use the stable API from hardware or applications:

```python
from recognizer_api import Recognizer

recognizer = Recognizer()
result = recognizer.predict(frame)  # frame shape: (16, 16)
```

See `API_DOCUMENT.md` and `example.py` in the project root.

Launch the local CSV posture recognition desktop app:

```bash
python3 posture_csv_app.py
```

This now loads `v2_2_candidate` by default. To run a historical version:

```bash
python3 posture_csv_app.py --model-version v1
python3 posture_csv_app.py --model-version v2_candidate
python3 posture_csv_app.py --model-version v2_1_candidate
python3 posture_csv_app.py --model-version v2_2_candidate
```

The GUI title, model information panel, and exported `summary.json` record the
actual runtime model version. V2.2 is displayed as
`V2.2（H3闭卷通过）`.

`v2_2_candidate` keeps V2.1 as the parent recognizer and adds a two-stage
leanback fine subclassifier for `后仰靠背坐` vs `后靠/瘫坐类`. If fine
subdivision is unsafe, it falls back to `后靠坐姿`. It was promoted after the
H3 external holdout: correct_accept 3/4, correct_fallback 1/4, wrong_accept 0,
gate_miss 0. H3 is consumed holdout evidence and must not enter training or
tuning.

Required runtime dependencies:

- Python 3
- `numpy`
- `scikit-learn`
- `joblib`
- `tkinter` with the local Python installation
- existing RF artifacts under `recognizer/models/`

The CSV app currently supports FlexPressureVision CSV playback only. It does
not connect to realtime hardware yet and does not retrain or modify the frozen
RF V1 model. In the app:

1. Click `选择CSV`.
2. Pick a FlexPressureVision CSV file.
3. Click `开始`.
4. Watch the 16x16 pressure heatmap, occupancy state, posture, confidence,
   second label, margin, boundary flag, and history table.
5. Click `导出结果` to write:
   - `frame_predictions.csv`
   - `posture_segments.csv`
   - `summary.json`

Exports are saved under `recognizer/gui_outputs/<csv_name_timestamp>/`.
They include model traceability fields in `frame_predictions.csv`,
`posture_segments.csv`, and `summary.json`:

- `model_version`
- `model_artifact_sha256`
- `metadata_sha256`
- `runtime_config_sha256`

## Current Model Status

V2.2 is the current default runtime model after H3 external holdout promotion.
It preserves the V2.1 parent model and adds the leanback two-stage fine
classifier:

- H3 valid files: 4/4
- correct_accept: 3/4
- correct_fallback: 1/4
- wrong_accept: 0/4
- gate_miss: 0/4
- safe resolution: 4/4

H3 is consumed external holdout evidence. It must not be used for training,
tuning, prototype construction, threshold selection, or future unopened holdout
claims.

### Previous Phase 1 Status

V2.1 candidate passed holdout_batch_02 Phase 1 closed-book testing:

- Boundary-aware file accuracy: 10/12 = 83.33%
- V2 candidate on the same batch: 75.00%
- V1 on the same batch: 16.67%
- 端正坐姿: 2/3
- 前倾端坐: 3/3
- 标准靠背坐: 2/2
- 交叉腿靠背坐: 1/2
- 盘腿坐: 2/2
- wrong accepted files: 0
- object pressure entering posture model: 0

Limitation: V2.2 has passed the H3 leanback fine-class closed-book test, but any
future change to the V2.2 gate, boundary, prototypes, or classifier requires a
new unopened H4 or later holdout batch. Keep V1, V2 candidate, and V2.1
available as rollback versions.

Run tests:

```bash
python3 -m unittest recognizer.tests.test_recognizer_core
```

Compare the practical V1 backends and save a prototype bank:

```bash
python3 -m recognizer.training --output-dir recognizer/outputs --models prototype,random_forest --save-prototype-bank
```

Freeze the RF V1 candidate:

```bash
python3 -m recognizer.model_artifact
```

Replay one CSV through the realtime RF V1 chain:

```bash
python3 -m recognizer.realtime_cli path/to/file.csv \
  --model recognizer/models/rf_posture_v1.joblib \
  --prototype-bank recognizer/models/prototype_bank_v1.json
```

Replay one CSV through occupancy + realtime posture recognition:

```bash
python3 -m recognizer.realtime_cli path/to/file.csv \
  --model recognizer/models/rf_posture_v1.joblib \
  --prototype-bank recognizer/models/prototype_bank_v1.json
```

## Occupancy & Human Verification

Current implemented states:

- `EMPTY`: no detectable pressure, or only zero/noise-level sparse response.
- `LOAD_BELOW_THRESHOLD`: repeatable weak nonzero sensor response exists, but it is below reliable occupancy thresholds.
- `OBJECT`: compact, concentrated, mostly static pressure candidate.
- `HUMAN`: broad, continuous pressure that resembles hip/thigh support or a human loading/unloading transition.
- `UNKNOWN`: occupied but conflicting or insufficient evidence.

Posture recognition is blocked unless occupancy is `HUMAN` and the seat window is stable. This prevents obvious object pressure from going directly into the RF posture model.

Hardware limitation: if a light object produces no numeric response and no stable nonzero 16x16 pressure region, software cannot detect it. That case is equivalent to `EMPTY`; it is not an OBJECT recognition failure.

Current detection thresholds in `OccupancyDetector`:

- `detectable_value_threshold = 1.0` per cell
- `detectable_total_threshold = 10.0`
- `detectable_points_min = 4`
- `occupied_total_threshold = 250.0`
- `active_value_threshold = 15.0` per cell

`LOAD_BELOW_THRESHOLD` requires a repeatable weak response: at least 4 cells above `1.0`, total pressure at least `10.0`, repeated over at least 3 frames, while still below the reliable occupancy threshold.

Minimum detectable object data to collect separately from the frozen posture dataset:

- `object_backpack_loaded_01.csv`, `object_backpack_loaded_02.csv`
- `object_books_heavy_01.csv`, `object_books_heavy_02.csv`
- `object_box_loaded_01.csv`, `object_box_loaded_02.csv`
- `object_heavy_single_01.csv`, `object_heavy_single_02.csv`

Each file should use: empty 2s -> place object -> hold 10s -> remove object -> empty 2s.

Before recording, manually confirm the object creates a stable nonzero pressure map. Do not record objects with no sensor response.

## Hardware Reader Contract

Realtime hardware only needs to provide:

```python
class FrameReader:
    def read_frame(self) -> np.ndarray:  # shape (16, 16)
        ...
```

`SerialFrameReader` is intentionally a skeleton until the hardware protocol is known. Required protocol fields:

- `port`
- `baudrate`
- `rows`, `cols`
- `delimiter`
- whether timestamps are included
- frame header / footer, if any
- orientation mapping: normal, flip left-right, flip up-down, rotate 180, transpose

Run an optional lightweight XGBoost baseline:

```bash
python3 -m recognizer.training --output-dir recognizer/outputs_xgboost --models xgboost
```

Replay one CSV:

```bash
python3 -m recognizer.predict path/to/file.csv --model recognizer/outputs/prototype_bank_v1.json
```

Open the simple GUI with CSV playback:

```bash
python3 -m recognizer.gui --model recognizer/outputs/prototype_bank_v1.json --csv path/to/file.csv
```
