# Recognizer API

This is the stable hardware-facing API for the 16x16 pressure-cushion recognizer.

Hardware code only needs to provide one realtime pressure frame:

```python
from recognizer_api import Recognizer

recognizer = Recognizer()
frame = read_frame()  # numpy array, shape=(16, 16)
result = recognizer.predict(frame)
```

The API hides Occupancy Detection, Human/Object gating, stable-window buffering,
feature extraction, Random Forest posture recognition, prototype diagnosis, and
time smoothing.

## Recognizer()

```python
recognizer = Recognizer()
```

Default initialization automatically loads:

- the model version pointed to by `recognizer/models/default_model.json`
- current default: `v2_2_candidate`
- `recognizer/models/rf_posture_v2_1_candidate.joblib`
- `recognizer/models/prototype_bank_v2_1_candidate.json`
- `recognizer/models/rf_posture_v2_1_candidate.metadata.json`
- `recognizer/models/rf_posture_v2_1_candidate.runtime_config.json`
- `recognizer/models/leanback_subclassifier_v2_2_candidate.joblib`
- `recognizer/models/leanback_prototype_bank_v2_2_candidate.json`
- `recognizer/models/leanback_subclassifier_v2_2_candidate.runtime_config.json`

Optional parameters are available for testing or advanced integration:

```python
Recognizer(
    model_version=None,  # None means use recognizer/models/default_model.json
    model_path=None,
    prototype_bank_path=None,
    metadata_path=None,
    runtime_config_path=None,
    fps=20.0,
    window_seconds=1.5,
    settle_seconds=1.0,
)
```

In normal hardware integration, use `Recognizer()` with no arguments.

Explicit historical versions remain available:

```python
Recognizer(model_version="v1")
Recognizer(model_version="v2_candidate")
Recognizer(model_version="v2_1_candidate")
Recognizer(model_version="v2_2_candidate")
```

The local CSV GUI uses the same version names:

```bash
python3 posture_csv_app.py                  # default: v2_2_candidate
python3 posture_csv_app.py --model-version v1
python3 posture_csv_app.py --model-version v2_candidate
python3 posture_csv_app.py --model-version v2_1_candidate
python3 posture_csv_app.py --model-version v2_2_candidate
```

`v2_2_candidate` keeps V2.1 as the parent model and adds a second-stage
leanback fine classifier. It is the current default after passing H3 external
holdout. Unsafe fine subdivision falls back to `后靠坐姿`.

## predict(frame)

```python
result = recognizer.predict(frame)
```

Input:

- `frame`: one pressure matrix as a NumPy-compatible array.
- Required shape: `(16, 16)`.
- Values should be numeric sensor pressure values after the hardware reader has
  applied any required parsing or orientation correction.

Invalid shapes raise `ValueError`.

Output:

```python
{
    "occupancy": "EMPTY|LOAD_BELOW_THRESHOLD|OBJECT|HUMAN|UNKNOWN",
    "occupancy_confidence": 0.0,
    "seat_state": "EMPTY|LOAD_BELOW_THRESHOLD|OBJECT|UNKNOWN|HUMAN_STABILIZING|HUMAN_RECOGNIZING",
    "posture": None,
    "posture_confidence": None,
    "second_label": None,
    "margin": None,
    "is_boundary": False,
    "prototype_diagnosis": None,
    "parent_posture_label": None,
    "fine_posture_label": None,
    "final_display_label": None,
    "subclassifier_triggered": None,
    "fine_boundary": None,
    "fallback_used": None,
    "reason": "...",
    "occupancy_features": {...}
}
```

Rules:

- `EMPTY`: no posture is returned.
- `LOAD_BELOW_THRESHOLD`: weak detectable load, no posture is returned.
- `OBJECT`: no posture is returned.
- `UNKNOWN`: no posture is returned.
- `HUMAN_STABILIZING`: human detected, waiting for stable window, no posture yet.
- `HUMAN_RECOGNIZING`: stable human window, posture recognition is allowed.

`posture` is only populated after the system has detected HUMAN occupancy and
the pressure window is stable.

## Boundary And Confidence

`posture_confidence` is the Random Forest confidence for the selected posture.

`second_label` is the second most likely posture.

`margin` is the confidence gap between the best and second label.

`is_boundary` is true when the recognizer considers the frame/window uncertain,
for example low confidence, small margin, or prototype diagnosis conflict. In
that case the UI should display an uncertain or boundary state instead of
treating the posture as fully reliable.

`prototype_diagnosis` is an explanation aid. It may contain the nearest
prototype label, margin, matched prototype id, and whether it agrees with the RF
model. It does not override the RF model by itself.

## reset()

```python
recognizer.reset()
```

Use this when:

- the GUI reset button is pressed;
- the user leaves the seat and the application wants a clean state;
- hardware reconnects and the realtime buffers should be cleared.

`reset()` clears smoothing, stable-window buffers, and previous posture state.

## calibrate()

```python
recognizer.calibrate()
recognizer.calibrate(frame=empty_frame)
recognizer.calibrate(frames=empty_frames)
```

Use this when the cushion is known to be empty and you want to refresh the
baseline.

- `calibrate()` clears the runtime analyzer and starts fresh baseline collection
  on future frames.
- `calibrate(frame=...)` uses one empty frame.
- `calibrate(frames=...)` uses a stack shaped `(n, 16, 16)`.

Do not call `calibrate()` while a person or object is on the cushion.

## Hardware Integration Contract

Hardware only needs to implement:

```python
def read_frame():
    # Return one numpy array shaped (16, 16)
    ...
```

Then call:

```python
recognizer = Recognizer()

while True:
    frame = read_frame()
    result = recognizer.predict(frame)
    print(result)
```

The hardware side does not need to load the Random Forest model, build features,
run occupancy detection, run prototype matching, or implement smoothing.

## Current Scope

The API is stable for integration with a realtime 16x16 pressure matrix stream.
The default model is currently V2.2 after H3 external holdout promotion:

- correct_accept: 3/4
- correct_fallback: 1/4
- wrong_accept: 0/4
- gate_miss: 0/4
- safe_resolution: 4/4

H3 is consumed external holdout evidence for this promotion. It must not enter
training, tuning, prototype construction, or future unopened holdout claims.
