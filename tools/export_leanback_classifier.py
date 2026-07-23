#!/usr/bin/env python3
"""Export the V2.4.3 runtime leanback fine classifier as portable JSON."""

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

from recognizer.leanback_subclassifier import (
    FINE_BOUNDARY_LABEL,
    FINE_LEANBACK_LABEL,
    FINE_SLOUCH_LABEL,
    LEANBACK_FEATURE_NAMES,
    LEANBACK_RELATED_LABELS,
    load_leanback_fine_model,
)
from recognizer.recognizer_api import Recognizer


SCHEMA_VERSION = "leanback_classifier_export_v1"
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


def load_runtime_leanback_model(model_version: str = DEFAULT_MODEL_VERSION) -> tuple[Recognizer, object]:
    recognizer = Recognizer(model_version=model_version)
    if recognizer.submodel_path is None:
        raise ValueError(f"model_version {model_version!r} does not define a leanback submodel")
    return recognizer, load_leanback_fine_model(recognizer.submodel_path)


def build_leanback_classifier_export(model_version: str = DEFAULT_MODEL_VERSION) -> dict[str, object]:
    recognizer, model = load_runtime_leanback_model(model_version)
    feature_names = list(model.feature_names or LEANBACK_FEATURE_NAMES)
    if len(feature_names) != len(model.feature_mean):
        raise ValueError("leanback feature_names and feature_mean dimensions do not match")
    runtime_config = None
    if recognizer.subruntime_config_path is not None and Path(recognizer.subruntime_config_path).exists():
        runtime_config = json.loads(Path(recognizer.subruntime_config_path).read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "submodel_version": model.submodel_version,
        "source_joblib": str(recognizer.submodel_path),
        "source_joblib_sha256": sha256_file(recognizer.submodel_path),
        "source_prototype_bank": None if recognizer.subprototype_bank_path is None else str(recognizer.subprototype_bank_path),
        "source_prototype_bank_sha256": None
        if recognizer.subprototype_bank_path is None
        else sha256_file(recognizer.subprototype_bank_path),
        "source_runtime_config": None if recognizer.subruntime_config_path is None else str(recognizer.subruntime_config_path),
        "source_runtime_config_sha256": None
        if recognizer.subruntime_config_path is None
        else sha256_file(recognizer.subruntime_config_path),
        "runtime_config_snapshot": runtime_config,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "joblib_version": __import__("joblib").__version__,
        "object_type": object_type_name(model),
        "classifier": None if model.classifier is None else {"type": object_type_name(model.classifier)},
        "selected_method": "prototype" if model.classifier is None else "classifier_plus_prototype",
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "feature_extractor": {
            "function": "recognizer.leanback_subclassifier.extract_leanback_features",
            "input": "16x16 frame or (n,16,16) window; window_average is used before feature extraction",
            "active_threshold": 15.0,
            "feature_order": feature_names,
            "scale_rule": "(feature - feature_mean) / feature_scale, abs(scale)<1e-9 replaced by 1.0",
        },
        "classes": [FINE_LEANBACK_LABEL, FINE_SLOUCH_LABEL],
        "fallback_label": FINE_BOUNDARY_LABEL,
        "parent_related_labels": sorted(LEANBACK_RELATED_LABELS),
        "gate": {
            "candidate_sources": ["parent.label", "parent.raw_label", "parent.prototype_diagnosis.label"],
            "parent_related_labels": [FINE_LEANBACK_LABEL, FINE_SLOUCH_LABEL],
            "physical_gate": {
                "left_share_min": 0.40,
                "left_share_max": 0.62,
                "row_0_3_plus_row_4_7_min": 0.72,
                "cop_y_min": 2.40,
                "cop_y_max": 5.80,
            },
        },
        "prototypes": {str(label): np.asarray(vector, dtype=np.float64).tolist() for label, vector in model.prototypes.items()},
        "feature_mean": np.asarray(model.feature_mean, dtype=np.float64).tolist(),
        "feature_scale": np.asarray(model.feature_scale, dtype=np.float64).tolist(),
        "distance_metric": "euclidean_l2_on_scaled_22_feature",
        "distance_thresholds": {str(key): float(value) for key, value in (model.distance_thresholds or {}).items()},
        "margin_threshold": float(model.margin_threshold),
        "confidence_threshold": float(model.confidence_threshold),
        "decision_rules": {
            "prototype_margin": "second_distance - best_distance",
            "prototype_confidence": "1 / (1 + best_distance) when classifier is None",
            "fallback_when": [
                "prototype_margin < margin_threshold",
                "best_distance > distance_thresholds[best_label]",
                "classifier_margin < margin_threshold when classifier exists",
                "classifier_confidence < confidence_threshold when classifier exists",
                "classifier/prototype conflict near boundary",
                "unknown fine label",
            ],
            "fallback_label": FINE_BOUNDARY_LABEL,
            "output_confidence_round_decimals": 6,
            "output_margin_round_decimals": 6,
            "output_distance_round_decimals": 6,
            "fine_boundary_reasons_sorted_unique": True,
            "top_level_confidence": "TwoStageLeanbackRecognizer keeps parent confidence after fine trigger",
            "top_level_boundary": "fine fallback clears top-level boundary; otherwise parent boundary is preserved",
        },
    }


def export_leanback_classifier(model_version: str, output_path: str | Path) -> dict[str, object]:
    payload = build_leanback_classifier_export(model_version)
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
    payload = export_leanback_classifier(args.model_version, args.output)
    print(f"Exported leanback classifier {payload['submodel_version']} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
