#!/usr/bin/env python3
"""Export parent-hybrid prototype/boundary golden records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, TOOLS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from recognizer.data_loader import read_sensor_csv
from recognizer.feature_extractor import FEATURE_DIM, FEATURE_NAMES, extract_features
from export_main_classifier import load_main_classifier
from export_parent_hybrid_model import DEFAULT_MODEL_VERSION, load_parent_hybrid


SCHEMA_VERSION = "parent_hybrid_golden_v1"


def _sha256_float64(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _argsort_desc(values: np.ndarray) -> list[int]:
    return [int(item) for item in np.argsort(values)[::-1]]


def _prototype_debug(parent: object, frame: np.ndarray, features: np.ndarray) -> dict[str, object]:
    prototype_recognizer = parent.prototype_recognizer
    mirror_feature = prototype_recognizer._mirror_feature(frame)
    distances, best_prototypes = prototype_recognizer._class_distances(features, mirror_feature)
    ordered = sorted(distances.items(), key=lambda item: item[1])
    best_label, best_distance = ordered[0]
    second_label, second_distance = ordered[1] if len(ordered) > 1 else (best_label, best_distance)
    margin = float(second_distance - best_distance)
    confidence = 1.0 if second_distance <= 0 else float(np.clip(1.0 - best_distance / second_distance, 0.0, 1.0))
    is_boundary = prototype_recognizer._is_boundary(best_label, best_distance, margin, confidence)
    public = prototype_recognizer.predict_posture(frame)
    bank = prototype_recognizer.prototype_bank
    threshold = bank.class_thresholds.get(best_label)
    class_distances = [
        {
            "label": label,
            "distance": float(distances[label]),
            "matched_prototype_id": best_prototypes[label].prototype_id,
        }
        for label in bank.labels
    ]
    return {
        "entered": True,
        "input_feature_sha256": _sha256_float64(features),
        "standardized_feature_sha256": _sha256_float64(bank.standardized(features)),
        "class_distances": class_distances,
        "best_label": best_label,
        "best_distance": float(best_distance),
        "second_label": second_label,
        "second_distance": float(second_distance),
        "margin": margin,
        "confidence": confidence,
        "class_threshold": None if threshold is None else float(threshold),
        "margin_threshold": float(prototype_recognizer.config.boundary_margin),
        "min_confidence": float(prototype_recognizer.config.min_confidence),
        "is_boundary": bool(is_boundary),
        "accepted": not bool(is_boundary),
        "matched_prototype_id": best_prototypes[best_label].prototype_id,
        "public_rounded": {
            "label": public.label,
            "confidence": public.confidence,
            "second_label": public.second_label,
            "margin": public.margin,
            "is_boundary": public.is_boundary,
            "best_distance": public.best_distance,
            "second_distance": public.second_distance,
            "matched_prototype_id": public.matched_prototype_id,
        },
        "overrides_main_classifier": False,
    }


def export_parent_hybrid_golden(
    pressure_frames_path: str | Path,
    output_path: str | Path,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> int:
    _, frames = read_sensor_csv(pressure_frames_path)
    _, classifier = load_main_classifier(model_version)
    _, parent = load_parent_hybrid(model_version)
    classes = [str(item) for item in classifier.classes_]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for frame_index, frame in enumerate(frames):
            features = np.asarray(extract_features(frame), dtype=np.float64)
            main_proba = np.asarray(classifier.predict_proba(features.reshape(1, -1)), dtype=float)[0]
            order = _argsort_desc(main_proba)
            main_label = classes[order[0]]
            second_label = classes[order[1]]
            main_confidence = float(main_proba[order[0]])
            main_margin = float(main_proba[order[0]] - main_proba[order[1]])
            rf = parent.rf_recognizer.predict_posture(frame)
            prototype = _prototype_debug(parent, frame, features)
            hybrid = parent.predict_posture(frame)
            record = {
                "schema_version": SCHEMA_VERSION,
                "frame_index": frame_index,
                "model_version": model_version,
                "feature_dim": FEATURE_DIM,
                "feature_names": list(FEATURE_NAMES),
                "features": features.tolist(),
                "feature_float64_sha256": _sha256_float64(features),
                "classes": classes,
                "main_classifier_predict_proba": main_proba.tolist(),
                "main_classifier_argmax_index": int(order[0]),
                "main_classifier_second_index": int(order[1]),
                "main_classifier_label": main_label,
                "main_classifier_confidence_raw": main_confidence,
                "main_classifier_second_label": second_label,
                "main_classifier_margin_raw": main_margin,
                "rf_probability_prediction": {
                    "label": rf.label,
                    "confidence": rf.confidence,
                    "second_label": rf.second_label,
                    "margin": rf.margin,
                    "is_boundary": rf.is_boundary,
                    "best_distance": rf.best_distance,
                    "second_distance": rf.second_distance,
                },
                "prototype": prototype,
                "boundary_metrics": {
                    "rf_confidence": rf.confidence,
                    "rf_margin": rf.margin,
                    "rf_is_boundary": rf.is_boundary,
                    "rf_confidence_threshold": parent.rf_recognizer.min_confidence,
                    "rf_margin_threshold": parent.rf_recognizer.boundary_margin,
                    "prototype_is_boundary": prototype["public_rounded"]["is_boundary"],
                    "prototype_label": prototype["public_rounded"]["label"],
                    "prototype_conflict": bool(
                        prototype["public_rounded"]["label"] != rf.label
                        and rf.margin <= parent.prototype_conflict_margin
                    ),
                    "prototype_boundary": bool(
                        prototype["public_rounded"]["is_boundary"]
                        and (
                            prototype["public_rounded"]["label"] != rf.label
                            or rf.confidence < parent.prototype_boundary_confidence_gate
                        )
                    ),
                    "prototype_conflict_margin": parent.prototype_conflict_margin,
                    "prototype_boundary_confidence_gate": parent.prototype_boundary_confidence_gate,
                },
                "boundary_reasons": list(hybrid.get("boundary_reasons") or []),
                "boundary_before_label": rf.label,
                "boundary_after_label": hybrid.get("label"),
                "boundary_before_confidence": rf.confidence,
                "boundary_after_confidence": hybrid.get("confidence"),
                "boundary_result": bool(hybrid.get("is_boundary")),
                "parent_hybrid_label": hybrid.get("label"),
                "parent_hybrid_confidence": hybrid.get("confidence"),
                "parent_hybrid_second_label": hybrid.get("second_label"),
                "parent_hybrid_margin": hybrid.get("margin"),
                "parent_hybrid_prototype_diagnosis": hybrid.get("prototype_diagnosis"),
                "actual_execution_branch": "parent_hybrid_only",
                "requires_leanback_subclassifier": False,
                "requires_lateral_subclassifier": False,
                "downstream_gate_note": "not evaluated before this stage boundary",
            }
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":")))
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
    count = export_parent_hybrid_golden(
        args.pressure_frames,
        args.output,
        model_version=args.model_version,
    )
    print(f"Exported {count} parent hybrid golden records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
