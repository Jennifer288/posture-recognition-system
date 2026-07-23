#!/usr/bin/env python3
"""Export the V2.4.3 parent CalibratedClassifierCV as portable JSON."""

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

from recognizer.feature_extractor import FEATURE_DIM, FEATURE_NAMES
from recognizer.model_artifact import load_model_bundle
from recognizer.recognizer_api import Recognizer


SCHEMA_VERSION = "main_classifier_export_v1"
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


def load_main_classifier(model_version: str = DEFAULT_MODEL_VERSION) -> tuple[Recognizer, object]:
    recognizer = Recognizer(model_version=model_version)
    bundle = load_model_bundle(recognizer.model_path)
    return recognizer, bundle["model"]


def _export_tree(tree: object, tree_index: int) -> dict[str, object]:
    tree_state = tree.tree_
    value = np.asarray(tree_state.value[:, 0, :], dtype=float)
    missing = getattr(tree_state, "missing_go_to_left", None)
    payload = {
        "tree_index": tree_index,
        "estimator_type": object_type_name(tree),
        "node_count": int(tree_state.node_count),
        "n_features": int(tree_state.n_features),
        "n_outputs": int(tree_state.n_outputs),
        "n_classes": [int(item) for item in np.atleast_1d(tree_state.n_classes)],
        "classes": [str(item) for item in np.asarray(tree.classes_)],
        "children_left": np.asarray(tree_state.children_left, dtype=np.int64).tolist(),
        "children_right": np.asarray(tree_state.children_right, dtype=np.int64).tolist(),
        "feature": np.asarray(tree_state.feature, dtype=np.int64).tolist(),
        "threshold": np.asarray(tree_state.threshold, dtype=np.float64).tolist(),
        "value": value.tolist(),
        "impurity": np.asarray(tree_state.impurity, dtype=np.float64).tolist(),
        "n_node_samples": np.asarray(tree_state.n_node_samples, dtype=np.int64).tolist(),
        "weighted_n_node_samples": np.asarray(tree_state.weighted_n_node_samples, dtype=np.float64).tolist(),
        "value_semantics": "sklearn tree_.value probabilities returned by DecisionTreeClassifier.predict_proba",
    }
    if missing is not None:
        payload["missing_go_to_left"] = np.asarray(missing, dtype=np.uint8).astype(int).tolist()
    return payload


def _export_calibrator(calibrator: object, class_index: int, class_label: str) -> dict[str, object]:
    kind = object_type_name(calibrator)
    payload: dict[str, object] = {
        "class_index": class_index,
        "class_label": class_label,
        "type": kind,
    }
    if kind.endswith("._SigmoidCalibration"):
        payload.update(
            {
                "method": "sigmoid",
                "a": float(calibrator.a_),
                "b": float(calibrator.b_),
                "formula": "expit(-(a * raw_prediction + b))",
            }
        )
    elif kind.endswith(".IsotonicRegression"):
        payload.update(
            {
                "method": "isotonic",
                "x_thresholds": np.asarray(calibrator.X_thresholds_, dtype=np.float64).tolist(),
                "y_thresholds": np.asarray(calibrator.y_thresholds_, dtype=np.float64).tolist(),
                "out_of_bounds": getattr(calibrator, "out_of_bounds", None),
            }
        )
    else:
        raise TypeError(f"Unsupported calibrator type: {kind}")
    return payload


def _export_calibrated_classifier(calibrated_classifier: object, fold_index: int, classes: list[str]) -> dict[str, object]:
    estimator = calibrated_classifier.estimator
    estimator_type = object_type_name(estimator)
    if not estimator_type.endswith(".RandomForestClassifier"):
        raise TypeError(f"Unsupported calibrated estimator type: {estimator_type}")

    estimators = list(estimator.estimators_)
    calibrators = list(calibrated_classifier.calibrators)
    return {
        "fold_index": fold_index,
        "type": object_type_name(calibrated_classifier),
        "classes": [str(item) for item in calibrated_classifier.classes],
        "estimator_type": estimator_type,
        "estimator_response_method": "predict_proba",
        "tree_input_dtype": "float32",
        "tree_branch_rule": "if isnan(x[feature]) use missing_go_to_left else x[feature] <= threshold goes left",
        "random_forest_params": json_safe(estimator.get_params()),
        "n_features_in": int(estimator.n_features_in_),
        "tree_count": len(estimators),
        "forest_probability_rule": "mean of DecisionTreeClassifier.predict_proba outputs in estimator.classes_ order",
        "calibrators": [
            _export_calibrator(calibrator, index, classes[index])
            for index, calibrator in enumerate(calibrators)
        ],
        "calibrated_probability_rule": (
            "apply one-vs-rest calibrator to each raw class probability, then normalize by row sum; "
            "if row sum is zero use uniform probability"
        ),
        "trees": [_export_tree(tree, index) for index, tree in enumerate(estimators)],
    }


def build_main_classifier_export(model_version: str = DEFAULT_MODEL_VERSION) -> dict[str, object]:
    recognizer, classifier = load_main_classifier(model_version)
    classifier_type = object_type_name(classifier)
    if not classifier_type.endswith(".CalibratedClassifierCV"):
        raise TypeError(f"Unsupported main classifier type: {classifier_type}")

    classes = [str(item) for item in classifier.classes_]
    calibrated = list(classifier.calibrated_classifiers_)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "source_joblib": str(recognizer.model_path),
        "source_joblib_sha256": sha256_file(recognizer.model_path),
        "runtime_artifacts": recognizer.artifact_identity(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "joblib_version": __import__("joblib").__version__,
        "classifier_type": classifier_type,
        "classes": classes,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "calibration_method": str(classifier.method),
        "ensemble": str(classifier.ensemble),
        "effective_ensemble_count": len(calibrated),
        "n_features_in": int(classifier.n_features_in_),
        "predict_proba_aggregation": "arithmetic mean of calibrated_classifiers_ predict_proba outputs",
        "probability_normalization": "per calibrated classifier normalize calibrated one-vs-rest outputs by row sum",
        "calibrated_classifiers": [
            _export_calibrated_classifier(item, index, classes)
            for index, item in enumerate(calibrated)
        ],
    }


def export_main_classifier(model_version: str | Path, output_path: str | Path | None = None) -> dict[str, object]:
    """Build and write a portable JSON model export.

    The first argument is intentionally named ``model_version`` for direct API use.
    """

    if output_path is None:
        raise ValueError("output_path is required")
    payload = build_main_classifier_export(str(model_version))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = export_main_classifier(args.model_version, args.output)
    print(f"Exported {payload['effective_ensemble_count']} calibrated classifiers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
