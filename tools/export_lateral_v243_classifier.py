#!/usr/bin/env python3
"""Export the V2.4.3 lateral merged fine model as portable JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognizer.lateral_merged_subclassifier import (
    DIAGONAL_SITTING_LABEL,
    LATERAL_UNCERTAIN_LABEL,
    SIDE_SITTING_OR_LEANING_LABEL,
)
from recognizer.lateral_merged_subclassifier_v243 import (
    CROSS_LEG_BACK_LABEL,
    LateralMergedFineModelV243,
    load_lateral_merged_fine_model_v243,
)
from recognizer.lateral_subclassifier import (
    DIAGONAL_SITTING_LABEL as ORIGINAL_DIAGONAL_SITTING_LABEL,
    LATERAL_FEATURE_NAMES,
    SIDE_LEANING_LABEL,
    STANDARD_SIDE_SITTING_LABEL,
)
from recognizer.recognizer_api import Recognizer


SCHEMA_VERSION = "lateral_v243_classifier_export_v1"
DEFAULT_MODEL_VERSION = "v2_4_3_candidate"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_type_name(obj: object) -> str:
    return type(obj).__module__ + "." + type(obj).__name__


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_runtime_lateral_model(model_version: str = DEFAULT_MODEL_VERSION) -> tuple[Recognizer, LateralMergedFineModelV243]:
    recognizer = Recognizer(model_version=model_version)
    if recognizer.lateral_submodel_path is None:
        raise ValueError(f"model_version {model_version!r} does not define a lateral submodel")
    return recognizer, load_lateral_merged_fine_model_v243(recognizer.lateral_submodel_path)


def _optional_json(path: str | Path | None) -> Any:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def build_lateral_v243_classifier_export(model_version: str = DEFAULT_MODEL_VERSION) -> dict[str, object]:
    recognizer, model = load_runtime_lateral_model(model_version)
    feature_names = list(model.feature_names or LATERAL_FEATURE_NAMES)
    if len(feature_names) != len(model.feature_mean):
        raise ValueError("lateral feature_names and feature_mean dimensions do not match")
    runtime_config = _optional_json(recognizer.lateral_runtime_config_path)
    metadata = _optional_json(getattr(recognizer, "lateral_metadata_path", None))
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = None
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "submodel_version": model.submodel_version,
        "source_joblib": str(recognizer.lateral_submodel_path),
        "source_joblib_sha256": sha256_file(recognizer.lateral_submodel_path),
        "source_prototype_bank": None if recognizer.lateral_prototype_bank_path is None else str(recognizer.lateral_prototype_bank_path),
        "source_prototype_bank_sha256": None
        if recognizer.lateral_prototype_bank_path is None
        else sha256_file(recognizer.lateral_prototype_bank_path),
        "source_runtime_config": None if recognizer.lateral_runtime_config_path is None else str(recognizer.lateral_runtime_config_path),
        "source_runtime_config_sha256": None
        if recognizer.lateral_runtime_config_path is None
        else sha256_file(recognizer.lateral_runtime_config_path),
        "runtime_config_snapshot": runtime_config,
        "metadata_snapshot": metadata,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn_version,
        "joblib_version": __import__("joblib").__version__,
        "object_type": object_type_name(model),
        "classifier": None if model.classifier is None else {"type": object_type_name(model.classifier)},
        "selected_method": "prototype" if model.classifier is None else "classifier_plus_prototype",
        "classes": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
        "fallback_label": LATERAL_UNCERTAIN_LABEL,
        "label_taxonomy_version": "lateral_merge_v1",
        "original_to_merged_label_mapping": {
            STANDARD_SIDE_SITTING_LABEL: SIDE_SITTING_OR_LEANING_LABEL,
            SIDE_LEANING_LABEL: SIDE_SITTING_OR_LEANING_LABEL,
            ORIGINAL_DIAGONAL_SITTING_LABEL: DIAGONAL_SITTING_LABEL,
        },
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "feature_extractor": {
            "function": "recognizer.lateral_subclassifier.extract_lateral_features",
            "input": "16x16 frame or (n,16,16) window; window_average is used before feature extraction",
            "active_threshold": 15.0,
            "feature_order": feature_names,
            "scale_rule": "(feature - feature_mean) / feature_scale, abs(scale)<1e-9 replaced by 1.0",
        },
        "prototypes": {
            str(label): [np.asarray(proto, dtype=np.float64).tolist() for proto in protos]
            for label, protos in model.prototypes.items()
        },
        "prototype_sources": model.prototype_sources,
        "prototype_subtypes": model.prototype_subtypes,
        "feature_mean": np.asarray(model.feature_mean, dtype=np.float64).tolist(),
        "feature_scale": np.asarray(model.feature_scale, dtype=np.float64).tolist(),
        "distance_metric": "euclidean_l2_on_scaled_42_feature",
        "class_distance_centers": {str(key): float(value) for key, value in (model.class_distance_centers or {}).items()},
        "class_distance_scales": {str(key): float(value) for key, value in (model.class_distance_scales or {}).items()},
        "margin_thresholds": {str(key): float(value) for key, value in (model.margin_thresholds or {}).items()},
        "distance_z_thresholds": {str(key): float(value) for key, value in (model.distance_z_thresholds or {}).items()},
        "confidence_threshold": float(model.confidence_threshold),
        "gate": {
            "candidate_labels": [STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL, ORIGINAL_DIAGONAL_SITTING_LABEL],
            "merged_output_labels": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
            "fallback_label": LATERAL_UNCERTAIN_LABEL,
            "candidate_sources": ["label", "final_display_label", "parent_posture_label", "raw_label", "second_label", "prototype_diagnosis.label"],
            "physical_gate": {
                "active_area_ratio_min": 0.22,
                "left_right_abs_balance_min": 0.10,
                "front_share_min": 0.18,
                "back_share_min": 0.28,
                "row_4_7_share_front_middle_dominant_max": 0.50,
                "row_8_11_share_back_mid_min_when_front_middle_dominant": 0.25,
                "row_8_11_share_min_when_cop_y_low": 0.26,
                "cop_y_low_threshold": 6.10,
            },
            "soft_warning_reasons": ["physical_gate_front_back_support_out_of_range"],
            "conditional_temporal_confirmation_frames": 3,
            "lateral_hold_frames": 6,
            "cross_leg_lateral_competition": {
                "enabled": True,
                "parent_label": CROSS_LEG_BACK_LABEL,
                "requires": [
                    "physical_ok",
                    "standard_side_signature",
                    "nearest_lateral_prototype_side",
                    "local_classifier_accepts_side",
                    "not_cross_leg_seat_signature",
                    "lateral_support_score - cross_leg_support_score >= 1.0",
                    "candidate_lateral_frames >= 3",
                ],
            },
        },
        "decision_rules": {
            "prototype_distance": "nearest prototype per class after scaling; sorted by distance using Python insertion order for ties",
            "second_label": "nearest different class; same class + 1.0 only when no different class exists",
            "distance_z": "max(0, (best_distance - class_distance_center[label]) / max(class_distance_scale[label], 1e-6))",
            "prototype_confidence": "1 / (1 + max(distance_z, 0.0)) when classifier is None",
            "fallback_when": [
                "distance_z > distance_z_thresholds[best_label]",
                "prototype_margin < margin_thresholds[best_label]",
                "classifier/prototype conflict when classifier exists",
                "low classifier/prototype confidence",
                "side_vs_diagonal_overlap",
                "unknown_lateral_label",
            ],
            "fallback_label": LATERAL_UNCERTAIN_LABEL,
            "output_confidence_round_decimals": 6,
            "output_margin_round_decimals": 6,
            "output_distance_round_decimals": 6,
            "fine_boundary_reasons_sorted_unique": True,
            "top_level_confidence": "TwoStageLateralMergedRecognizerV243 uses max(parent confidence, 0.72) after lateral resolution",
            "top_level_margin": "TwoStageLateralMergedRecognizerV243 uses max(parent margin, 0.20) after lateral resolution",
            "top_level_boundary": "lateral resolution clears parent boundary; lateral fallback emits 侧向姿势 with is_boundary=False",
        },
    }


def export_lateral_v243_classifier(model_version: str, output_path: str | Path) -> dict[str, object]:
    payload = build_lateral_v243_classifier_export(model_version)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = export_lateral_v243_classifier(args.model_version, args.output)
    print(f"Exported lateral V2.4.3 classifier {payload['submodel_version']} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
