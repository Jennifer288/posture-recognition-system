#!/usr/bin/env python3
"""Export leanback-stage golden records for V2.4.3."""

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
from recognizer.leanback_subclassifier import (
    LEANBACK_FEATURE_NAMES,
    extract_leanback_features,
    should_run_leanback_subclassifier,
)
from export_leanback_classifier import DEFAULT_MODEL_VERSION, load_runtime_leanback_model
from export_main_classifier import load_main_classifier
from export_parent_hybrid_model import load_parent_hybrid


SCHEMA_VERSION = "leanback_golden_v1"


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


def _fine_debug(model: object, feature_vector: np.ndarray) -> dict[str, object]:
    vector = np.asarray(feature_vector, dtype=float).reshape(-1)
    scaled = model._scale(vector)
    distances = {
        label: float(np.linalg.norm(scaled - model._scale(proto)))
        for label, proto in model.prototypes.items()
    }
    ordered = sorted(distances.items(), key=lambda item: item[1])
    best_label, best_distance = ordered[0]
    second_label, second_distance = ordered[1] if len(ordered) > 1 else ("", best_distance + 1.0)
    prototype_margin = float(second_distance - best_distance)
    prototype_confidence = float(1.0 / (1.0 + best_distance))
    result = model.predict_from_features(feature_vector)
    return {
        "scaled_features": scaled.tolist(),
        "scaled_feature_sha256": _sha256_float64(scaled),
        "class_distances": [
            {"label": label, "distance": float(distances[label])}
            for label in model.prototypes.keys()
        ],
        "best_label": best_label,
        "best_distance": float(best_distance),
        "second_label": second_label,
        "second_distance": float(second_distance),
        "prototype_margin_raw": prototype_margin,
        "prototype_confidence_raw": prototype_confidence,
        "classifier_present": model.classifier is not None,
        "predict_proba": None,
        "argmax_index": list(model.prototypes.keys()).index(best_label),
        "raw_label": best_label,
        "raw_confidence": prototype_confidence,
        "raw_margin": prototype_margin,
        "accepted": not bool(result.get("fallback_used")),
        "rejection_reasons": list(result.get("fine_boundary_reasons") or []),
        "result": result,
    }


def export_leanback_golden(
    pressure_frames_path: str | Path,
    output_path: str | Path,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> int:
    _, frames = read_sensor_csv(pressure_frames_path)
    _, main_classifier = load_main_classifier(model_version)
    _, parent = load_parent_hybrid(model_version)
    _, fine_model = load_runtime_leanback_model(model_version)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    class_order = list(fine_model.prototypes.keys())
    feature_names = list(fine_model.feature_names or LEANBACK_FEATURE_NAMES)

    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for frame_index, frame in enumerate(frames):
            frame_uint8 = np.asarray(frame, dtype=np.uint8)
            parent_features = np.asarray(extract_features(frame), dtype=np.float64)
            main_proba = np.asarray(main_classifier.predict_proba(parent_features.reshape(1, -1)), dtype=float)[0]
            parent_result = dict(parent.predict_posture(frame))
            feature_vector, feature_map = extract_leanback_features(frame)
            should_run, gate_reasons = should_run_leanback_subclassifier(parent_result, feature_map)
            fine_debug = _fine_debug(fine_model, feature_vector)
            payload = dict(parent_result)
            payload.update(
                {
                    "parent_posture_label": parent_result.get("label"),
                    "fine_posture_label": None,
                    "final_display_label": parent_result.get("label"),
                    "subclassifier_triggered": should_run,
                    "subclassifier_gate_reason": "; ".join(gate_reasons) if gate_reasons else "",
                    "fine_confidence": None,
                    "fine_margin": None,
                    "fine_boundary": False,
                    "fine_boundary_reasons": [],
                    "fine_prototype_label": None,
                    "fine_prototype_distance": None,
                    "fallback_used": False,
                    "model_version": "v2_2_candidate",
                    "parent_model_version": "v2_1_candidate",
                    "submodel_version": fine_model.submodel_version,
                }
            )
            if should_run:
                fine = dict(fine_debug["result"])
                fine["fine_feature_summary"] = feature_map
                payload.update(fine)
                payload["label"] = fine["final_display_label"]
                payload["confidence"] = parent_result.get("confidence", fine["fine_confidence"])
                payload["second_label"] = fine.get("fine_second_label") or parent_result.get("second_label")
                payload["margin"] = parent_result.get("margin", fine["fine_margin"])
                payload["is_boundary"] = bool(parent_result.get("is_boundary", False)) and not fine.get("fallback_used", False)
                if fine.get("fallback_used"):
                    payload["is_boundary"] = False
                payload["boundary_reasons"] = list(parent_result.get("boundary_reasons") or [])
                payload["fine_boundary_reasons"] = list(fine.get("fine_boundary_reasons") or [])

            record = {
                "schema_version": SCHEMA_VERSION,
                "frame_index": frame_index,
                "model_version": model_version,
                "frame_uint8": frame_uint8.reshape(-1).tolist(),
                "feature_dim": FEATURE_DIM,
                "feature_names": list(FEATURE_NAMES),
                "features": parent_features.tolist(),
                "feature_float64_sha256": _sha256_float64(parent_features),
                "classes": [str(item) for item in main_classifier.classes_],
                "main_classifier_predict_proba": main_proba.tolist(),
                "parent_hybrid": {
                    "label": parent_result.get("label"),
                    "confidence": parent_result.get("confidence"),
                    "second_label": parent_result.get("second_label"),
                    "margin": parent_result.get("margin"),
                    "is_boundary": parent_result.get("is_boundary"),
                    "boundary_reasons": parent_result.get("boundary_reasons") or [],
                    "prototype_diagnosis": parent_result.get("prototype_diagnosis"),
                },
                "leanback_triggered": bool(should_run),
                "leanback_trigger_reasons": list(gate_reasons),
                "leanback_gate_reason": "; ".join(gate_reasons) if gate_reasons else "",
                "leanback_feature_names": feature_names,
                "leanback_features": np.asarray(feature_vector, dtype=np.float64).tolist(),
                "leanback_feature_sha256": _sha256_float64(np.asarray(feature_vector, dtype=np.float64)),
                "leanback_feature_summary": feature_map,
                "leanback_classes": class_order,
                "leanback_predict_proba": fine_debug["predict_proba"] if should_run else None,
                "leanback_argmax_index": fine_debug["argmax_index"] if should_run else None,
                "leanback_raw_label": fine_debug["raw_label"] if should_run else None,
                "leanback_raw_confidence": fine_debug["raw_confidence"] if should_run else None,
                "leanback_raw_margin": fine_debug["raw_margin"] if should_run else None,
                "leanback_class_distances": fine_debug["class_distances"] if should_run else [],
                "leanback_scaled_features": fine_debug["scaled_features"] if should_run else [],
                "leanback_accepted": bool(fine_debug["accepted"]) if should_run else False,
                "leanback_rejection_reason": "; ".join(fine_debug["rejection_reasons"]) if should_run else "",
                "leanback_output_label": payload.get("label"),
                "leanback_output_confidence": payload.get("confidence"),
                "leanback_output_boundary": payload.get("is_boundary"),
                "leanback_fine_confidence": payload.get("fine_confidence"),
                "leanback_fine_margin": payload.get("fine_margin"),
                "leanback_fine_boundary": payload.get("fine_boundary"),
                "leanback_fine_boundary_reasons": payload.get("fine_boundary_reasons"),
                "leanback_fine_prototype_label": payload.get("fine_prototype_label"),
                "leanback_fine_prototype_distance": payload.get("fine_prototype_distance"),
                "leanback_fallback_used": payload.get("fallback_used"),
                "should_enter_lateral_stage": not bool(should_run),
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
    count = export_leanback_golden(args.pressure_frames, args.output, model_version=args.model_version)
    print(f"Exported {count} leanback golden records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
