#!/usr/bin/env python3
"""Export Python main-classifier-only golden predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, TOOLS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from recognizer.data_loader import read_sensor_csv
from recognizer.feature_extractor import FEATURE_DIM, FEATURE_NAMES, extract_features
from export_main_classifier import DEFAULT_MODEL_VERSION, load_main_classifier, object_type_name


SCHEMA_VERSION = "main_classifier_golden_v1"


def _sha256_float64(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def _sigmoid_calibrator_output(calibrator: object, raw_prediction: float) -> float:
    return float(calibrator.predict(np.asarray([raw_prediction], dtype=float))[0])


def _fold_payload(calibrated_classifier: object, features: np.ndarray, fold_index: int) -> dict[str, object]:
    estimator = calibrated_classifier.estimator
    raw = np.asarray(estimator.predict_proba(features.reshape(1, -1)), dtype=float)[0]
    calibrator_outputs = [
        _sigmoid_calibrator_output(calibrator, float(raw[index]))
        for index, calibrator in enumerate(calibrated_classifier.calibrators)
    ]
    calibrated = np.asarray(calibrated_classifier.predict_proba(features.reshape(1, -1)), dtype=float)[0]
    return {
        "fold_index": fold_index,
        "estimator_type": object_type_name(estimator),
        "estimator_proba": raw.tolist(),
        "calibrator_outputs_before_normalization": calibrator_outputs,
        "normalized_proba": calibrated.tolist(),
    }


def export_main_classifier_golden(
    pressure_frames_path: str | Path,
    output_path: str | Path,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> int:
    _, frames = read_sensor_csv(pressure_frames_path)
    _, classifier = load_main_classifier(model_version)
    classes = [str(item) for item in classifier.classes_]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for frame_index, frame in enumerate(frames):
            features = np.asarray(extract_features(frame), dtype=np.float64)
            final_proba = np.asarray(classifier.predict_proba(features.reshape(1, -1)), dtype=float)[0]
            best_index = int(np.argmax(final_proba))
            record = {
                "schema_version": SCHEMA_VERSION,
                "frame_index": frame_index,
                "model_version": model_version,
                "feature_dim": FEATURE_DIM,
                "feature_names": list(FEATURE_NAMES),
                "features": features.tolist(),
                "feature_float64_sha256": _sha256_float64(features),
                "classes": classes,
                "folds": [
                    _fold_payload(item, features, index)
                    for index, item in enumerate(classifier.calibrated_classifiers_)
                ],
                "final_predict_proba": final_proba.tolist(),
                "argmax_class_index": best_index,
                "main_classifier_prediction": classes[best_index],
                "main_classifier_confidence": float(final_proba[best_index]),
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-frames", required=True, type=Path)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    count = export_main_classifier_golden(
        args.pressure_frames,
        args.output,
        model_version=args.model_version,
    )
    print(f"Exported {count} main classifier golden records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
