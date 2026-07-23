#!/usr/bin/env python3
"""Export V2.4.3 parent prototype/boundary configuration as portable JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognizer.recognizer_api import Recognizer
from recognizer.rf_recognizer import HybridPostureRecognizer


SCHEMA_VERSION = "parent_hybrid_model_v1"
DEFAULT_MODEL_VERSION = "v2_4_3_candidate"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def find_parent_hybrid(runtime: object) -> HybridPostureRecognizer:
    current = runtime
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, HybridPostureRecognizer):
            return current
        visited.add(id(current))
        current = getattr(current, "parent_recognizer", None)
    raise TypeError("Could not locate HybridPostureRecognizer in runtime chain")


def load_parent_hybrid(model_version: str = DEFAULT_MODEL_VERSION) -> tuple[Recognizer, HybridPostureRecognizer]:
    recognizer = Recognizer(model_version=model_version)
    return recognizer, find_parent_hybrid(recognizer._posture_recognizer)


def build_parent_hybrid_export(model_version: str = DEFAULT_MODEL_VERSION) -> dict[str, object]:
    recognizer, parent = load_parent_hybrid(model_version)
    prototype_recognizer = parent.prototype_recognizer
    if prototype_recognizer is None:
        raise ValueError("Parent hybrid recognizer does not have a prototype recognizer")
    bank = prototype_recognizer.prototype_bank

    prototypes = [
        {
            "prototype_id": prototype.prototype_id,
            "label": prototype.label,
            "vector": np.asarray(prototype.vector, dtype=np.float64).tolist(),
            "mirror_aware": bool(prototype.mirror_aware),
            "source_files": list(prototype.source_files),
            "source_group": prototype.source_group,
        }
        for prototype in bank.prototypes
    ]
    runtime_config = None
    if recognizer.runtime_config_path is not None and recognizer.runtime_config_path.exists():
        runtime_config = json.loads(recognizer.runtime_config_path.read_text(encoding="utf-8"))

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "parent_stage_definition": (
            "main CalibratedClassifierCV predict_proba -> PrototypeRecognizer diagnosis -> "
            "HybridPostureRecognizer boundary flags; before leanback/lateral/time smoothing"
        ),
        "feature_dim": int(bank.feature_mean.shape[0]),
        "source_files": {
            "prototype_bank": str(recognizer.prototype_bank_path),
            "prototype_bank_sha256": sha256_file(recognizer.prototype_bank_path),
            "runtime_config": None if recognizer.runtime_config_path is None else str(recognizer.runtime_config_path),
            "runtime_config_sha256": None
            if recognizer.runtime_config_path is None
            else sha256_file(recognizer.runtime_config_path),
            "model_bundle": None if recognizer.model_bundle_path is None else str(recognizer.model_bundle_path),
            "model_bundle_sha256": None
            if recognizer.model_bundle_path is None
            else sha256_file(recognizer.model_bundle_path),
        },
        "runtime_config_snapshot": runtime_config,
        "rf_boundary": {
            "min_confidence": float(parent.rf_recognizer.min_confidence),
            "boundary_margin": float(parent.rf_recognizer.boundary_margin),
            "rf_prediction_round_decimals": 4,
            "rf_is_boundary_rule": "raw confidence < min_confidence or raw margin < boundary_margin",
            "hybrid_reason_rule": "rounded RF confidence/margin are compared again inside HybridPostureRecognizer",
        },
        "prototype_bank": {
            "version": "recognizer_v1_prototype_bank",
            "feature_dim": int(bank.feature_mean.shape[0]),
            "labels": list(bank.labels),
            "feature_mean": np.asarray(bank.feature_mean, dtype=np.float64).tolist(),
            "feature_std": np.asarray(bank.feature_std, dtype=np.float64).tolist(),
            "class_thresholds": {str(key): float(value) for key, value in bank.class_thresholds.items()},
            "margin_threshold": float(bank.margin_threshold),
            "label_taxonomy": dict(bank.label_taxonomy),
            "prototypes": prototypes,
        },
        "prototype_distance": {
            "input": "extract_features(window)",
            "space": "standardized_264_feature",
            "standardization": "(feature - feature_mean) / feature_std, with zero std replaced by 1.0 at load time",
            "metric": "euclidean_l2",
            "mirror_rule": "if prototype.mirror_aware, use min(distance(feature), distance(mirror_feature)); current bank has no mirror-aware prototypes",
            "class_distance_rule": "minimum distance among prototypes for that label",
            "class_order": "PrototypeBank.labels = sorted unique prototype labels; Python sort is stable for equal distances",
        },
        "prototype_decision": {
            "confidence_rule": "1.0 if second_distance <= 0 else clip(1.0 - best_distance / second_distance, 0.0, 1.0)",
            "confidence_round_decimals": 4,
            "distance_round_decimals": 4,
            "margin_round_decimals": 4,
            "boundary_rules": [
                "best_distance > class_threshold[label]",
                "margin < margin_threshold",
                "confidence < RecognizerConfig.min_confidence",
            ],
            "use_bank_thresholds": bool(prototype_recognizer.config.use_bank_thresholds),
            "config_min_confidence": float(prototype_recognizer.config.min_confidence),
            "margin_threshold": float(prototype_recognizer.config.boundary_margin),
            "prototype_override_parent_label": False,
        },
        "hybrid_boundary": {
            "prototype_conflict_margin": float(parent.prototype_conflict_margin),
            "prototype_boundary_confidence_gate": float(parent.prototype_boundary_confidence_gate),
            "prototype_boundary_rule": (
                "proto.is_boundary and (proto.label != rf.label or "
                "rounded_rf_confidence < prototype_boundary_confidence_gate)"
            ),
            "prototype_conflict_rule": "proto.label != rf.label and rounded_rf_margin <= prototype_conflict_margin",
            "final_label_rule": "keep RF label; prototype only adds diagnosis/boundary reasons",
            "final_confidence_rule": "keep rounded RF confidence",
            "boundary_reason_order": [
                "RF confidence",
                "RF margin",
                "Prototype boundary",
                "Prototype/RF conflict",
            ],
        },
        "downstream": {
            "leanback_subclassifier": "not evaluated in this parent hybrid export",
            "lateral_subclassifier": "not evaluated in this parent hybrid export",
            "time_smoothing": "not evaluated in this parent hybrid export",
        },
    }


def export_parent_hybrid_model(model_version: str, output_path: str | Path) -> dict[str, object]:
    payload = build_parent_hybrid_export(model_version)
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
    payload = export_parent_hybrid_model(args.model_version, args.output)
    print(
        f"Exported {len(payload['prototype_bank']['prototypes'])} parent prototypes "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
