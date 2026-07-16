from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .data_loader import DEFAULT_FORMAL_DATASET, WindowSample
from .feature_extractor import FEATURE_DIM
from .training import ModelEvaluation, _stack_xy, build_v1_prototype_bank, evaluate_random_forest_leave_one_csv_out, load_default_samples


FEATURE_VERSION = "pressure_features_v1_264"
RF_RANDOM_SEED = 42
RF_PARAMS = {
    "n_estimators": 40,
    "random_state": RF_RANDOM_SEED,
    "class_weight": "balanced_subsample",
    "min_samples_leaf": 2,
    "n_jobs": 1,
}


def create_rf_model() -> object:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**RF_PARAMS)


def train_rf_model(samples: Sequence[WindowSample]) -> object:
    x_train, y_train = _stack_xy(samples)
    model = create_rf_model()
    model.fit(x_train, y_train)
    return model


def save_model_bundle(path: Path | str, model: object, metadata: dict[str, object]) -> None:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, target)


def load_model_bundle(path: Path | str) -> dict[str, object]:
    import joblib

    return joblib.load(path)


def build_metadata(
    samples: Sequence[WindowSample],
    validation: ModelEvaluation,
    model_path: Path,
    prototype_path: Path,
) -> dict[str, object]:
    labels = sorted({sample.label for sample in samples})
    return {
        "model_name": "rf_posture_v1",
        "model_role": "Random Forest realtime primary recognizer",
        "dataset_version": "dataset_v1_1_17_final + V1 evaluation-layer labels for classes 8/10/11",
        "formal_dataset_path": str(DEFAULT_FORMAL_DATASET),
        "training_csv_count": len(samples),
        "labels": labels,
        "label_count": len(labels),
        "feature_dim": FEATURE_DIM,
        "feature_version": FEATURE_VERSION,
        "feature_extractor": "recognizer.feature_extractor.extract_features",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RF_RANDOM_SEED,
        "rf_params": RF_PARAMS,
        "validation": {
            "method": "Leave-One-Independent-CSV-Out",
            "file_accuracy": validation.file_accuracy,
            "window_accuracy": validation.window_accuracy,
            "boundary_rate": validation.boundary_rate,
            "per_class_file_recall": validation.per_class_file_recall,
            "confusion_matrix": validation.confusion_matrix,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": dependency_versions(),
        "artifacts": {
            "rf_model": str(model_path),
            "prototype_bank": str(prototype_path),
        },
    }


def dependency_versions() -> dict[str, str]:
    versions = {"numpy": np.__version__}
    for module_name in ["sklearn", "joblib"]:
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - environment detail
            versions[module_name] = f"unavailable: {exc}"
    return versions


def save_rf_v1_candidate(output_dir: Path | str = "recognizer/models") -> dict[str, Path | bool]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "rf_posture_v1.joblib"
    metadata_path = target / "rf_posture_v1.metadata.json"
    prototype_path = target / "prototype_bank_v1.json"

    samples = load_default_samples()
    validation = evaluate_random_forest_leave_one_csv_out(samples)
    model = train_rf_model(samples)
    prototype_bank = build_v1_prototype_bank(samples)
    prototype_bank.save(prototype_path)
    metadata = build_metadata(samples, validation, model_path=model_path, prototype_path=prototype_path)
    save_model_bundle(model_path, model=model, metadata=metadata)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    consistency = check_save_load_consistency(model_path, samples)
    metadata["save_load_consistent"] = consistency
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle = load_model_bundle(model_path)
    bundle["metadata"] = metadata
    save_model_bundle(model_path, model=bundle["model"], metadata=metadata)
    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "prototype_path": prototype_path,
        "save_load_consistent": consistency,
    }


def check_save_load_consistency(model_path: Path | str, samples: Sequence[WindowSample], max_windows: int = 128) -> bool:
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    features = np.vstack([sample.features for sample in samples])
    if len(features) > max_windows:
        features = features[:max_windows]
    before = model.predict(features)
    loaded = load_model_bundle(model_path)["model"]
    after = loaded.predict(features)
    return bool(np.array_equal(before, after))


def main() -> int:
    paths = save_rf_v1_candidate()
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
