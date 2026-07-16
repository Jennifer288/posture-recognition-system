from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data_loader import WindowSample, default_training_files, load_window_sample
from .prototype_bank import Prototype, PrototypeBank
from .recognizer import PrototypeRecognizer


MULTI_PROTOTYPE_BY_CSV = {"class5", "class11"}
SOURCE_GROUP_PROTOTYPES = {"class8", "class10"}


@dataclass(frozen=True)
class FilePrediction:
    model: str
    csv_file: str
    true_label: str
    predicted_label: str
    correct: bool
    window_count: int
    window_accuracy: float
    boundary_rate: float
    mean_confidence: float
    matched_prototypes: str = ""


@dataclass(frozen=True)
class ModelEvaluation:
    model: str
    file_accuracy: float
    window_accuracy: float
    boundary_rate: float
    per_class_file_recall: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    file_predictions: list[FilePrediction]
    status: str = "ok"
    detail: str = ""


def build_v1_prototype_bank(samples: Sequence[WindowSample]) -> PrototypeBank:
    if not samples:
        raise ValueError("At least one training sample is required")
    features = np.vstack([sample.features for sample in samples])
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    groups: dict[tuple[str, str, str], list[WindowSample]] = defaultdict(list)
    for sample in samples:
        family = sample.source_family
        if family in MULTI_PROTOTYPE_BY_CSV:
            key = (sample.label, family, sample.path.name)
        elif family in SOURCE_GROUP_PROTOTYPES:
            key = (sample.label, family, family)
        else:
            key = (sample.label, "label", sample.label)
        groups[key].append(sample)

    prototypes = []
    for (label, source_group, group_id), group_samples in sorted(groups.items(), key=lambda item: item[0]):
        group_features = np.vstack([sample.features for sample in group_samples])
        source_files = tuple(sorted(sample.path.name for sample in group_samples))
        prototype_id = f"{label}::{source_group}::{group_id}"
        prototypes.append(
            Prototype(
                prototype_id=prototype_id,
                label=label,
                vector=group_features.mean(axis=0),
                mirror_aware=source_group == "class11",
                source_files=source_files,
                source_group=source_group,
            )
        )

    bank = PrototypeBank(
        prototypes,
        feature_mean=feature_mean,
        feature_std=feature_std,
        label_taxonomy={
            "半躺靠背坐": "后靠/瘫坐类",
            "瘫坐/斜躺合并": "后靠/瘫坐类",
            "全躺卧姿": "躺卧类",
            "侧卧半躺": "躺卧类",
        },
    )
    class_thresholds, margin_threshold = _fit_boundary_thresholds(bank, samples)
    return PrototypeBank(
        prototypes,
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_thresholds=class_thresholds,
        margin_threshold=margin_threshold,
        label_taxonomy=bank.label_taxonomy,
    )


def evaluate_prototype_leave_one_csv_out(samples: Sequence[WindowSample]) -> ModelEvaluation:
    predictions: list[FilePrediction] = []
    total_windows = 0
    correct_windows = 0
    boundary_windows = 0
    for test_index, test_sample in enumerate(samples):
        train_samples = [sample for index, sample in enumerate(samples) if index != test_index]
        bank = build_v1_prototype_bank(train_samples)
        recognizer = PrototypeRecognizer(bank)
        window_labels = []
        confidences = []
        matched = []
        boundary_count = 0
        for window in test_sample.windows:
            result = recognizer.predict_posture(window)
            window_labels.append(result.label)
            confidences.append(result.confidence)
            matched.append(result.matched_prototype_id)
            boundary_count += int(result.is_boundary)
        predicted = _majority_label(window_labels)
        correct_count = sum(label == test_sample.label for label in window_labels)
        total_windows += len(window_labels)
        correct_windows += correct_count
        boundary_windows += boundary_count
        predictions.append(
            FilePrediction(
                model="Prototype Recognizer",
                csv_file=test_sample.path.name,
                true_label=test_sample.label,
                predicted_label=predicted,
                correct=predicted == test_sample.label,
                window_count=len(window_labels),
                window_accuracy=correct_count / max(len(window_labels), 1),
                boundary_rate=boundary_count / max(len(window_labels), 1),
                mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
                matched_prototypes=";".join(_top_items(matched, limit=3)),
            )
        )
    return _summarize_model("Prototype Recognizer", predictions, correct_windows, total_windows, boundary_windows)


def evaluate_random_forest_leave_one_csv_out(samples: Sequence[WindowSample]) -> ModelEvaluation:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as exc:  # pragma: no cover - depends on local environment
        return _unavailable("Random Forest", exc)

    def factory() -> object:
        return RandomForestClassifier(
            n_estimators=40,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=1,
        )

    return _evaluate_sklearn_model("Random Forest", samples, factory)


def evaluate_xgboost_leave_one_csv_out(samples: Sequence[WindowSample]) -> ModelEvaluation:
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:  # pragma: no cover - depends on local environment
        return _unavailable("XGBoost", exc)

    predictions: list[FilePrediction] = []
    total_windows = 0
    correct_windows = 0
    boundary_windows = 0
    for test_index, test_sample in enumerate(samples):
        train_samples = [sample for index, sample in enumerate(samples) if index != test_index]
        x_train, y_train = _stack_xy(train_samples)
        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(y_train)
        model = XGBClassifier(
            n_estimators=4,
            max_depth=2,
            learning_rate=0.12,
            subsample=0.85,
            colsample_bytree=0.80,
            tree_method="hist",
            max_bin=64,
            n_jobs=1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(x_train, encoded_y)
        proba = model.predict_proba(test_sample.features)
        predicted_indices = np.argmax(proba, axis=1)
        window_labels = list(encoder.inverse_transform(predicted_indices))
        confidences = proba.max(axis=1)
        margins = _probability_margins(proba)
        boundary_count = int(np.sum((confidences < 0.55) | (margins < 0.10)))
        correct_count = sum(label == test_sample.label for label in window_labels)
        total_windows += len(window_labels)
        correct_windows += correct_count
        boundary_windows += boundary_count
        predictions.append(
            FilePrediction(
                model="XGBoost",
                csv_file=test_sample.path.name,
                true_label=test_sample.label,
                predicted_label=_majority_label(window_labels),
                correct=_majority_label(window_labels) == test_sample.label,
                window_count=len(window_labels),
                window_accuracy=correct_count / max(len(window_labels), 1),
                boundary_rate=boundary_count / max(len(window_labels), 1),
                mean_confidence=float(np.mean(confidences)) if len(confidences) else 0.0,
            )
        )
    return _summarize_model("XGBoost", predictions, correct_windows, total_windows, boundary_windows)


def compare_v1_models(
    samples: Sequence[WindowSample],
    models: Sequence[str] = ("prototype", "random_forest", "xgboost"),
) -> list[ModelEvaluation]:
    evaluations = []
    for model in models:
        key = model.strip().lower()
        if key in {"prototype", "prototype_recognizer"}:
            evaluations.append(evaluate_prototype_leave_one_csv_out(samples))
        elif key in {"random_forest", "rf"}:
            evaluations.append(evaluate_random_forest_leave_one_csv_out(samples))
        elif key in {"xgboost", "xgb"}:
            evaluations.append(evaluate_xgboost_leave_one_csv_out(samples))
        else:
            evaluations.append(_unavailable(model, ValueError(f"Unknown model backend: {model}")))
    return evaluations


def write_evaluation_outputs(evaluations: Sequence[ModelEvaluation], output_dir: Path | str) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_csv = target / "model_comparison_summary.csv"
    predictions_csv = target / "model_comparison_predictions.csv"
    matrices_json = target / "model_comparison_confusion_matrices.json"
    summary_json = target / "model_comparison_summary.json"
    recommendation_md = target / "model_recommendation.md"

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "status", "file_accuracy", "window_accuracy", "boundary_rate", "detail"],
        )
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(
                {
                    "model": evaluation.model,
                    "status": evaluation.status,
                    "file_accuracy": f"{evaluation.file_accuracy:.6f}",
                    "window_accuracy": f"{evaluation.window_accuracy:.6f}",
                    "boundary_rate": f"{evaluation.boundary_rate:.6f}",
                    "detail": evaluation.detail,
                }
            )

    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "csv_file",
                "true_label",
                "predicted_label",
                "correct",
                "window_count",
                "window_accuracy",
                "boundary_rate",
                "mean_confidence",
                "matched_prototypes",
            ],
        )
        writer.writeheader()
        for evaluation in evaluations:
            for prediction in evaluation.file_predictions:
                writer.writerow(_file_prediction_row(prediction))

    matrices_json.write_text(
        json.dumps({evaluation.model: evaluation.confusion_matrix for evaluation in evaluations}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps([_evaluation_to_dict(evaluation) for evaluation in evaluations], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    recommendation_md.write_text(_recommendation(evaluations), encoding="utf-8")
    return {
        "summary_csv": summary_csv,
        "predictions_csv": predictions_csv,
        "confusion_matrices_json": matrices_json,
        "summary_json": summary_json,
        "recommendation_md": recommendation_md,
    }


def load_default_samples(window: int = 8, step: int = 2) -> list[WindowSample]:
    return [load_window_sample(path, window=window, step=step) for path in default_training_files()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Recognizer V1 model backends with CSV-grouped validation.")
    parser.add_argument("--output-dir", default="recognizer/outputs", help="Directory for comparison reports.")
    parser.add_argument("--window", type=int, default=8, help="Rolling window size in frames.")
    parser.add_argument("--step", type=int, default=2, help="Window step in frames.")
    parser.add_argument(
        "--models",
        default="prototype,random_forest,xgboost",
        help="Comma-separated backends: prototype, random_forest, xgboost.",
    )
    parser.add_argument("--save-prototype-bank", action="store_true", help="Also save a prototype bank trained on all current samples.")
    args = parser.parse_args(argv)

    samples = load_default_samples(window=args.window, step=args.step)
    evaluations = compare_v1_models(samples, models=tuple(item.strip() for item in args.models.split(",") if item.strip()))
    paths = write_evaluation_outputs(evaluations, args.output_dir)
    if args.save_prototype_bank:
        bank = build_v1_prototype_bank(samples)
        paths["prototype_bank"] = Path(args.output_dir) / "prototype_bank_v1.json"
        bank.save(paths["prototype_bank"])
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


def _fit_boundary_thresholds(bank: PrototypeBank, samples: Sequence[WindowSample]) -> tuple[dict[str, float], float]:
    recognizer = PrototypeRecognizer(bank)
    by_label: dict[str, list[float]] = defaultdict(list)
    margins: list[float] = []
    for sample in samples:
        for feature in sample.features:
            distances, _ = recognizer._class_distances(feature, feature)
            ordered = sorted(distances.items(), key=lambda item: item[1])
            by_label[sample.label].append(ordered[0][1])
            if len(ordered) > 1:
                margins.append(ordered[1][1] - ordered[0][1])
    thresholds = {}
    for label, values in by_label.items():
        arr = np.asarray(values, dtype=float)
        thresholds[label] = float(np.percentile(arr, 97.5) * 1.15 + 1e-6)
    margin_threshold = float(np.percentile(np.asarray(margins), 5) * 0.50) if margins else 0.0
    return thresholds, max(0.0, margin_threshold)


def _evaluate_sklearn_model(model_name: str, samples: Sequence[WindowSample], factory: object) -> ModelEvaluation:
    predictions: list[FilePrediction] = []
    total_windows = 0
    correct_windows = 0
    boundary_windows = 0
    for test_index, test_sample in enumerate(samples):
        train_samples = [sample for index, sample in enumerate(samples) if index != test_index]
        x_train, y_train = _stack_xy(train_samples)
        model = factory()
        model.fit(x_train, y_train)
        window_labels = list(model.predict(test_sample.features))
        boundary_count = 0
        mean_confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(test_sample.features)
            confidences = proba.max(axis=1)
            margins = _probability_margins(proba)
            boundary_count = int(np.sum((confidences < 0.55) | (margins < 0.10)))
            mean_confidence = float(np.mean(confidences))
        correct_count = sum(label == test_sample.label for label in window_labels)
        total_windows += len(window_labels)
        correct_windows += correct_count
        boundary_windows += boundary_count
        predicted = _majority_label(window_labels)
        predictions.append(
            FilePrediction(
                model=model_name,
                csv_file=test_sample.path.name,
                true_label=test_sample.label,
                predicted_label=predicted,
                correct=predicted == test_sample.label,
                window_count=len(window_labels),
                window_accuracy=correct_count / max(len(window_labels), 1),
                boundary_rate=boundary_count / max(len(window_labels), 1),
                mean_confidence=mean_confidence,
            )
        )
    return _summarize_model(model_name, predictions, correct_windows, total_windows, boundary_windows)


def _stack_xy(samples: Sequence[WindowSample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([sample.features for sample in samples])
    y = np.asarray([sample.label for sample in samples for _ in range(len(sample.features))])
    return x, y


def _probability_margins(proba: np.ndarray) -> np.ndarray:
    if proba.shape[1] < 2:
        return np.ones(proba.shape[0])
    sorted_proba = np.sort(proba, axis=1)
    return sorted_proba[:, -1] - sorted_proba[:, -2]


def _majority_label(labels: Sequence[str]) -> str:
    if not labels:
        return ""
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _top_items(items: Iterable[str], limit: int = 3) -> list[str]:
    counts = Counter(items)
    return [f"{item}:{count}" for item, count in counts.most_common(limit)]


def _summarize_model(
    model_name: str,
    predictions: list[FilePrediction],
    correct_windows: int,
    total_windows: int,
    boundary_windows: int,
) -> ModelEvaluation:
    labels = sorted({prediction.true_label for prediction in predictions} | {prediction.predicted_label for prediction in predictions})
    confusion = {label: {predicted: 0 for predicted in labels} for label in labels}
    by_class = defaultdict(list)
    for prediction in predictions:
        confusion[prediction.true_label][prediction.predicted_label] += 1
        by_class[prediction.true_label].append(prediction.correct)
    per_class = {label: float(np.mean(values)) for label, values in sorted(by_class.items())}
    return ModelEvaluation(
        model=model_name,
        file_accuracy=sum(prediction.correct for prediction in predictions) / max(len(predictions), 1),
        window_accuracy=correct_windows / max(total_windows, 1),
        boundary_rate=boundary_windows / max(total_windows, 1),
        per_class_file_recall=per_class,
        confusion_matrix=confusion,
        file_predictions=predictions,
    )


def _unavailable(model_name: str, exc: Exception) -> ModelEvaluation:
    return ModelEvaluation(
        model=model_name,
        file_accuracy=0.0,
        window_accuracy=0.0,
        boundary_rate=0.0,
        per_class_file_recall={},
        confusion_matrix={},
        file_predictions=[],
        status="unavailable",
        detail=str(exc),
    )


def _file_prediction_row(prediction: FilePrediction) -> dict[str, object]:
    return {
        "model": prediction.model,
        "csv_file": prediction.csv_file,
        "true_label": prediction.true_label,
        "predicted_label": prediction.predicted_label,
        "correct": int(prediction.correct),
        "window_count": prediction.window_count,
        "window_accuracy": f"{prediction.window_accuracy:.6f}",
        "boundary_rate": f"{prediction.boundary_rate:.6f}",
        "mean_confidence": f"{prediction.mean_confidence:.6f}",
        "matched_prototypes": prediction.matched_prototypes,
    }


def _evaluation_to_dict(evaluation: ModelEvaluation) -> dict[str, object]:
    return {
        "model": evaluation.model,
        "status": evaluation.status,
        "detail": evaluation.detail,
        "file_accuracy": evaluation.file_accuracy,
        "window_accuracy": evaluation.window_accuracy,
        "boundary_rate": evaluation.boundary_rate,
        "per_class_file_recall": evaluation.per_class_file_recall,
        "confusion_matrix": evaluation.confusion_matrix,
        "file_predictions": [_file_prediction_row(prediction) for prediction in evaluation.file_predictions],
    }


def _recommendation(evaluations: Sequence[ModelEvaluation]) -> str:
    available = [evaluation for evaluation in evaluations if evaluation.status == "ok"]
    if not available:
        return "# Recognizer V1 Recommendation\n\nNo model backend was available in this Python environment.\n"
    best = sorted(available, key=lambda item: (item.file_accuracy, item.window_accuracy, -item.boundary_rate), reverse=True)[0]
    lines = [
        "# Recognizer V1 Recommendation",
        "",
        f"Recommended backend: **{best.model}**",
        "",
        "| Model | Status | File Accuracy | Window Accuracy | Boundary Rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for evaluation in evaluations:
        lines.append(
            f"| {evaluation.model} | {evaluation.status} | {evaluation.file_accuracy:.4f} | "
            f"{evaluation.window_accuracy:.4f} | {evaluation.boundary_rate:.4f} |"
        )
    lines.extend(
        [
            "",
            "Selection rule: prefer the highest grouped file accuracy, then window accuracy, then lower boundary rate.",
            "The frozen CSV data is read-only; this script only writes comparison reports under the requested output directory.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
