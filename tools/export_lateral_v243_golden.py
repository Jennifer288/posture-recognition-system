#!/usr/bin/env python3
"""Export stateful V2.4.3 lateral-stage golden records."""

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
from recognizer.lateral_merged_subclassifier import (
    DIAGONAL_SITTING_LABEL,
    LATERAL_UNCERTAIN_LABEL,
    SIDE_SITTING_OR_LEANING_LABEL,
)
from recognizer.lateral_subclassifier import LATERAL_FEATURE_NAMES, extract_lateral_features
from recognizer.recognizer_api import Recognizer
from export_lateral_v243_classifier import DEFAULT_MODEL_VERSION
from export_main_classifier import load_main_classifier
from export_parent_hybrid_model import load_parent_hybrid


SCHEMA_VERSION = "lateral_v243_golden_v1"


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


def _split_reason_text(value: object) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _compact_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "label",
        "confidence",
        "second_label",
        "margin",
        "is_boundary",
        "boundary_reasons",
        "prototype_diagnosis",
        "parent_posture_label",
        "fine_posture_label",
        "subclassifier_triggered",
        "subclassifier_gate_reason",
        "fine_confidence",
        "fine_margin",
        "fine_boundary",
        "fine_boundary_reasons",
        "fine_prototype_label",
        "fine_prototype_distance",
        "fallback_used",
        "final_display_label",
    ]
    return {key: payload.get(key) for key in keys if key in payload}


def _lateral_debug(model: object, feature_vector: np.ndarray) -> dict[str, object]:
    vector = np.asarray(feature_vector, dtype=float).reshape(-1)
    scaled = model._scale(vector)
    ordered = model._prototype_distances(scaled)
    best_label, best_source, best_subtype, best_distance = ordered[0]
    second_label, second_source, second_subtype, second_distance = model._second_other_label(ordered, best_label)
    prototype_margin = float(second_distance - best_distance)
    distance_z = float(model._distance_z(best_label, best_distance))
    prototype_confidence = float(1.0 / (1.0 + max(distance_z, 0.0)))
    result = model.predict_from_features(feature_vector)
    return {
        "scaled_features": scaled.tolist(),
        "scaled_feature_sha256": _sha256_float64(scaled),
        "class_distances": [
            {
                "label": label,
                "source": source,
                "subtype": subtype,
                "distance": float(distance),
            }
            for label, source, subtype, distance in ordered
        ],
        "best_label": best_label,
        "best_source": best_source,
        "best_subtype": best_subtype,
        "best_distance": float(best_distance),
        "second_label": second_label,
        "second_source": second_source,
        "second_subtype": second_subtype,
        "second_distance": float(second_distance),
        "prototype_margin_raw": prototype_margin,
        "distance_z_raw": distance_z,
        "prototype_confidence_raw": prototype_confidence,
        "classifier_present": model.classifier is not None,
        "predict_proba": None,
        "argmax_index": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL].index(best_label),
        "raw_label": best_label,
        "raw_confidence": prototype_confidence,
        "raw_margin": prototype_margin,
        "accepted": not bool(result.get("lateral_fallback_used")),
        "rejection_reasons": list(result.get("lateral_boundary_reasons") or []),
        "result": result,
    }


def export_lateral_v243_golden(
    pressure_frames_path: str | Path,
    output_path: str | Path,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> int:
    _, frames = read_sensor_csv(pressure_frames_path)
    _, main_classifier = load_main_classifier(model_version)
    _, parent = load_parent_hybrid(model_version)
    recognizer = Recognizer(model_version=model_version)
    lateral_stage = recognizer._posture_recognizer
    lateral_stage.reset()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for frame_index, frame in enumerate(frames):
            frame_uint8 = np.asarray(frame, dtype=np.uint8)
            parent_features = np.asarray(extract_features(frame), dtype=np.float64)
            main_proba = np.asarray(main_classifier.predict_proba(parent_features.reshape(1, -1)), dtype=float)[0]
            parent_result = dict(parent.predict_posture(frame))
            leanback_result = dict(lateral_stage.parent_recognizer.predict_posture(frame))
            lateral_features, lateral_feature_map = extract_lateral_features(frame)
            lateral_debug = _lateral_debug(lateral_stage.lateral_model, lateral_features)
            output = dict(lateral_stage.predict_posture(frame))
            temporal_state = output.get("lateral_temporal_state") or "inactive"
            active_or_hold = bool(output.get("lateral_subclassifier_triggered"))
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
                "leanback_stage": _compact_prediction(leanback_result),
                "leanback_input_label": leanback_result.get("label"),
                "leanback_input_confidence": leanback_result.get("confidence"),
                "leanback_input_boundary": leanback_result.get("is_boundary"),
                "lateral_triggered": active_or_hold,
                "lateral_temporal_state": temporal_state,
                "lateral_trigger_reason": output.get("lateral_gate_reason") or "",
                "lateral_trigger_reasons": _split_reason_text(output.get("lateral_gate_reason")),
                "lateral_gate_metrics": {
                    "physical_features": lateral_feature_map,
                    "lateral_gate_candidate": output.get("lateral_gate_candidate"),
                    "lateral_physical_evidence_passed": output.get("lateral_physical_evidence_passed"),
                    "lateral_physical_evidence_reasons": output.get("lateral_physical_evidence_reasons"),
                    "lateral_gate_strong_evidence": output.get("lateral_gate_strong_evidence"),
                    "lateral_gate_soft_warnings": output.get("lateral_gate_soft_warnings"),
                    "lateral_gate_hard_reject_reasons": output.get("lateral_gate_hard_reject_reasons"),
                    "front_back_support_warning": output.get("front_back_support_warning"),
                    "front_back_support_hard_reject": output.get("front_back_support_hard_reject"),
                    "parent_prototype_agreement": output.get("parent_prototype_agreement"),
                    "cross_leg_lateral_competition_active": output.get("cross_leg_lateral_competition_active"),
                    "cross_leg_lateral_competition_reason": output.get("cross_leg_lateral_competition_reason"),
                    "cross_leg_support_score": output.get("cross_leg_support_score"),
                    "lateral_support_score": output.get("lateral_support_score"),
                    "lateral_vs_cross_leg_margin": output.get("lateral_vs_cross_leg_margin"),
                    "conditional_gate_override": output.get("conditional_gate_override"),
                    "conditional_gate_override_reason": output.get("conditional_gate_override_reason"),
                    "cross_leg_seat_protection_active": output.get("cross_leg_seat_protection_active"),
                    "cross_leg_seat_protection_reasons": output.get("cross_leg_seat_protection_reasons"),
                },
                "lateral_feature_names": list(lateral_stage.lateral_model.feature_names or LATERAL_FEATURE_NAMES),
                "lateral_features": np.asarray(lateral_features, dtype=np.float64).tolist(),
                "lateral_feature_sha256": _sha256_float64(np.asarray(lateral_features, dtype=np.float64)),
                "lateral_classes": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
                "lateral_raw_scores": None,
                "lateral_predict_proba": lateral_debug["predict_proba"],
                "lateral_class_distances": lateral_debug["class_distances"],
                "lateral_scaled_features": lateral_debug["scaled_features"],
                "lateral_argmax_index": lateral_debug["argmax_index"],
                "lateral_raw_label": lateral_debug["raw_label"],
                "lateral_raw_confidence": lateral_debug["raw_confidence"],
                "lateral_raw_margin": lateral_debug["raw_margin"],
                "lateral_distance_z": lateral_debug["distance_z_raw"],
                "lateral_accepted": bool(output.get("lateral_subclassifier_triggered")) and not bool(output.get("lateral_fallback_used")),
                "lateral_rejection_reason": output.get("fallback_reason") or "; ".join(output.get("lateral_boundary_reasons") or []),
                "lateral_fallback_used": output.get("lateral_fallback_used"),
                "lateral_output_label": output.get("label"),
                "lateral_output_confidence": output.get("confidence"),
                "lateral_output_boundary": output.get("is_boundary"),
                "lateral_output_raw_label": output.get("lateral_merged_label") or output.get("lateral_posture_label"),
                "lateral_output_second_label": output.get("second_label"),
                "lateral_output_margin": output.get("margin"),
                "selected_branch": output.get("selected_branch"),
                "final_selected_branch": output.get("final_selected_branch"),
                "final_priority_branch": output.get("final_priority_branch"),
                "smoothing_input": {
                    "label": output.get("label"),
                    "final_display_label": output.get("final_display_label"),
                    "confidence": output.get("confidence"),
                    "second_label": output.get("second_label"),
                    "margin": output.get("margin"),
                    "is_boundary": output.get("is_boundary"),
                    "boundary_reasons": output.get("boundary_reasons") or [],
                    "lateral_subclassifier_triggered": output.get("lateral_subclassifier_triggered"),
                    "lateral_temporal_state": output.get("lateral_temporal_state"),
                    "selected_branch": output.get("selected_branch"),
                    "final_priority_branch": output.get("final_priority_branch"),
                },
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
    count = export_lateral_v243_golden(args.pressure_frames, args.output, model_version=args.model_version)
    print(f"Exported {count} lateral V2.4.3 golden records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
