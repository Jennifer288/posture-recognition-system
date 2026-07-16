from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognizer.csv_gui_core import CsvRecognitionSession, load_csv_playback
from recognizer.data_loader import read_sensor_csv, stable_frames
from recognizer.feature_extractor import windowed_frames
from recognizer.lateral_subclassifier import (
    DIAGONAL_SITTING_LABEL,
    LATERAL_BOUNDARY_LABEL,
    LATERAL_FEATURE_NAMES,
    SIDE_LEANING_LABEL,
    STANDARD_SIDE_SITTING_LABEL,
    LateralFineModel,
    extract_lateral_features,
    save_lateral_fine_model,
    save_lateral_prototype_bank,
    should_run_lateral_subclassifier,
)
from recognizer.recognizer_api import Recognizer, default_model_version, sha256_file

OUTPUT_ROOT = PROJECT_ROOT / "posture_dataset_v2" / "v2_3_candidate"
REPORT_DIR = PROJECT_ROOT / "posture_dataset_v2" / "reports" / "v2_3_lateral_subclassifier"
PLOT_DIR = REPORT_DIR / "plots"
MODEL_DIR = PROJECT_ROOT / "recognizer" / "models"
SUBMODEL_PATH = MODEL_DIR / "lateral_subclassifier_v2_3_candidate.joblib"
SUBMODEL_METADATA_PATH = MODEL_DIR / "lateral_subclassifier_v2_3_candidate.metadata.json"
SUBMODEL_RUNTIME_CONFIG_PATH = MODEL_DIR / "lateral_subclassifier_v2_3_candidate.runtime_config.json"
SUBMODEL_PROTOTYPE_PATH = MODEL_DIR / "lateral_prototype_bank_v2_3_candidate.json"
MODEL_BUNDLE_PATH = MODEL_DIR / "v2_3_candidate.model_bundle.json"

DATA_FILES = {
    "SY1_ceshenyikaozuo1.csv": (SIDE_LEANING_LABEL, "SY1", "posture_dataset_v2/development/screening/side_leaning_sy1_raw/SY1_ceshenyikaozuo1.csv"),
    "SY1_ceshenyikaozuo2.csv": (SIDE_LEANING_LABEL, "SY1", "posture_dataset_v2/development/screening/side_leaning_sy1_raw/SY1_ceshenyikaozuo2.csv"),
    "SY2_ceshenyikaozuo1.csv": (SIDE_LEANING_LABEL, "SY2", "posture_dataset_v2/development/separability/side_leaning_sy2_raw/SY2_ceshenyikaozuo1.csv"),
    "SY2_ceshenyikaozuo2.csv": (SIDE_LEANING_LABEL, "SY2", "posture_dataset_v2/development/separability/side_leaning_sy2_raw/SY2_ceshenyikaozuo2.csv"),
    "XC1_xiekuazuo1.csv": (DIAGONAL_SITTING_LABEL, "XC1", "posture_dataset_v2/development/screening/diagonal_sitting_xc1_raw/XC1_xiekuazuo1.csv"),
    "XC1_xiekuazuo2.csv": (DIAGONAL_SITTING_LABEL, "XC1", "posture_dataset_v2/development/screening/diagonal_sitting_xc1_raw/XC1_xiekuazuo2.csv"),
    "CS1_biaozhuncezuo1.csv": (STANDARD_SIDE_SITTING_LABEL, "CS1", "posture_dataset_v2/development/screening/standard_side_sitting_cs1_raw/CS1_biaozhuncezuo1.csv"),
    "CS1_biaozhuncezuo2.csv": (STANDARD_SIDE_SITTING_LABEL, "CS1", "posture_dataset_v2/development/screening/standard_side_sitting_cs1_raw/CS1_biaozhuncezuo2.csv"),
}
BOUNDARY_FILES = {
    "SY1_ceshenyikaozuo2.csv": "side_leaning_diagonal_nearest_boundary",
    "XC1_xiekuazuo2.csv": "side_leaning_diagonal_nearest_boundary",
    "SY2_ceshenyikaozuo1.csv": "side_leaning_standard_side_nearest_boundary",
    "CS1_biaozhuncezuo2.csv": "side_leaning_standard_side_nearest_boundary",
}


@dataclass(frozen=True)
class LateralSample:
    filename: str
    path: Path
    true_label: str
    batch: str
    frames: np.ndarray
    stable: np.ndarray
    windows: np.ndarray
    features: np.ndarray
    feature_maps: list[dict[str, float]]
    stable_start: int
    stable_end: int
    quality_score: int
    validity: str
    quality_notes: str


def main() -> int:
    setup_dirs()
    before = artifact_hashes()
    samples = load_samples()
    manifest_rows = manifest(samples)
    write_csv(OUTPUT_ROOT / "v2_3_development_manifest.csv", manifest_rows)
    write_csv(REPORT_DIR / "v2_3_development_manifest.csv", manifest_rows)
    write_csv(REPORT_DIR / "lateral_gate_analysis.csv", lateral_gate_analysis(samples))
    write_csv(REPORT_DIR / "grouped_cv_splits.csv", split_rows(samples))

    methods = ["prototype", "logistic_regression", "linear_svm", "lda", "random_forest", "physical_hybrid"]
    comparison_rows: list[dict[str, Any]] = []
    lofo_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        rows = run_lofo(samples, method)
        lofo_by_method[method] = rows
        comparison_rows.append(summarize_method(method, rows))
    write_csv(REPORT_DIR / "candidate_model_comparison.csv", comparison_rows)
    selected_method = choose_method(comparison_rows)
    selected_lofo = lofo_by_method[selected_method]
    write_csv(REPORT_DIR / "lofo_file_results.csv", selected_lofo)

    sy1_sy2 = batch_side_transfer(samples, selected_method, train_side_batch="SY1", test_side_batch="SY2")
    sy2_sy1 = batch_side_transfer(samples, selected_method, train_side_batch="SY2", test_side_batch="SY1")
    write_csv(REPORT_DIR / "sy1_train_sy2_test_results.csv", sy1_sy2)
    write_csv(REPORT_DIR / "sy2_train_sy1_test_results.csv", sy2_sy1)
    write_csv(REPORT_DIR / "boundary_case_analysis.csv", boundary_case_rows(selected_lofo, sy1_sy2, sy2_sy1))

    final_model = fit_model(samples, selected_method)
    save_artifacts(samples, final_model, selected_method, comparison_rows, selected_lofo, sy1_sy2, sy2_sy1)
    write_csv(REPORT_DIR / "multiprototype_analysis.csv", multiprototype_rows(final_model))
    write_csv(REPORT_DIR / "feature_coefficients_or_importance.csv", feature_rows(final_model, samples))
    write_csv(REPORT_DIR / "boundary_threshold_report.csv", threshold_rows(final_model))

    # Runtime regressions after the candidate artifacts exist and are registered.
    gate_regression = lateral_gate_regression(samples)
    non_lateral = non_lateral_false_trigger_report()
    leanback_regression = leanback_regression_report()
    object_rows = object_empty_unknown_gate_report()
    write_csv(REPORT_DIR / "lateral_gate_regression_results.csv", gate_regression)
    write_csv(REPORT_DIR / "non_lateral_false_trigger_report.csv", non_lateral)
    write_csv(REPORT_DIR / "leanback_regression_report.csv", leanback_regression)
    write_csv(REPORT_DIR / "object_empty_unknown_gate_report.csv", object_rows)

    after = artifact_hashes()
    artifact_payload = artifact_manifest(before, after)
    write_json(REPORT_DIR / "v2_3_artifact_manifest.json", artifact_payload)
    generate_plots(samples, selected_lofo, final_model)
    write_report(samples, selected_method, comparison_rows, selected_lofo, sy1_sy2, sy2_sy1, non_lateral, leanback_regression, object_rows, artifact_payload)
    (REPORT_DIR / "tests_report.md").write_text("# V2.3 Tests\n\n自动化测试将在候选生成后由主流程运行并回填最终结果。\n", encoding="utf-8")

    print(json.dumps({
        "selected_method": selected_method,
        "lofo": summarize_method(selected_method, selected_lofo),
        "sy1_train_sy2": summarize_method("SY1->SY2", sy1_sy2),
        "sy2_train_sy1": summarize_method("SY2->SY1", sy2_sy1),
        "submodel_path": str(SUBMODEL_PATH),
        "report_dir": str(REPORT_DIR),
        "default_model": default_model_version(),
        "v2_2_hashes_unchanged": v22_hashes(before) == v22_hashes(after),
    }, ensure_ascii=False, indent=2))
    return 0


def setup_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_samples() -> list[LateralSample]:
    samples: list[LateralSample] = []
    for filename, (label, batch, relpath) in DATA_FILES.items():
        path = PROJECT_ROOT / relpath
        if not path.exists():
            raise FileNotFoundError(path)
        data = load_csv_playback(path)
        frames = data.frames
        start, end = stable_bounds(frames, data.fps)
        stable = frames[start:end]
        windows = windowed_frames(stable, window=8, step=2)
        feature_rows = []
        feature_maps = []
        for window in windows:
            vector, fmap = extract_lateral_features(window)
            feature_rows.append(vector)
            feature_maps.append(fmap)
        quality_score, validity, notes = quality_for(frames, stable, data.fps)
        samples.append(LateralSample(
            filename=filename,
            path=path,
            true_label=label,
            batch=batch,
            frames=frames,
            stable=stable,
            windows=windows,
            features=np.vstack(feature_rows),
            feature_maps=feature_maps,
            stable_start=start,
            stable_end=end,
            quality_score=quality_score,
            validity=validity,
            quality_notes=notes,
        ))
    return sorted(samples, key=lambda item: item.filename)


def stable_bounds(frames: np.ndarray, fps: float = 20.0) -> tuple[int, int]:
    totals = np.asarray(frames, dtype=float).sum(axis=(1, 2))
    if len(totals) == 0:
        return 0, 0
    p95 = float(np.percentile(totals, 95))
    occupied_threshold = max(250.0, p95 * 0.20)
    occupied = np.flatnonzero(totals >= occupied_threshold)
    if len(occupied) == 0:
        return 0, 0
    first = int(occupied[0])
    last = int(occupied[-1])
    trim = max(1, int(round(0.30 * fps)))
    start = min(first + trim, last)
    end = max(last - trim + 1, start + 1)
    return start, end


def quality_for(frames: np.ndarray, stable: np.ndarray, fps: float) -> tuple[int, str, str]:
    score = 100
    notes = []
    if len(stable) < max(10, int(round(fps * 2))):
        score -= 35
        notes.append("stable segment shorter than 2s")
    totals = frames.sum(axis=(1, 2)) if len(frames) else np.asarray([])
    stable_totals = stable.sum(axis=(1, 2)) if len(stable) else np.asarray([])
    if len(totals) == 0:
        return 0, "invalid", "empty or unreadable"
    front_empty = float(totals[: max(1, int(round(1.5 * fps)))].mean())
    back_empty = float(totals[-max(1, int(round(1.5 * fps))) :].mean())
    if front_empty > 250:
        score -= 8
        notes.append("front empty segment not fully empty")
    if back_empty > 300:
        score -= 8
        notes.append("tail empty segment not fully empty")
    if len(stable_totals):
        drift = abs(float(stable_totals[-1] - stable_totals[0])) / max(float(np.mean(stable_totals)), 1e-9)
        cv = float(np.std(stable_totals) / max(float(np.mean(stable_totals)), 1e-9))
        if drift > 0.25:
            score -= 12
            notes.append("stable total pressure drift >25%")
        if cv > 0.15:
            score -= 8
            notes.append("stable total pressure CV >15%")
    validity = "valid" if score >= 80 else "borderline_valid" if score >= 55 else "invalid"
    return max(0, int(score)), validity, "; ".join(notes) if notes else "ok"


def manifest(samples: list[LateralSample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        rows.append({
            "filename": sample.filename,
            "source_path": str(sample.path),
            "sha256": sha256_file(sample.path),
            "true_label": sample.true_label,
            "batch": sample.batch,
            "quality_score": sample.quality_score,
            "validity": sample.validity,
            "stable_start": sample.stable_start,
            "stable_end": sample.stable_end,
            "number_of_windows": len(sample.features),
            "data_role": "v2_3_lateral_development",
            "included_in_training": True,
            "included_in_validation": "grouped_only",
            "eligible_for_final_holdout": False,
            "notes": sample.quality_notes,
        })
    return rows


def build_training_matrix(samples: list[LateralSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x = np.vstack([sample.features for sample in samples])
    y = np.asarray([sample.true_label for sample in samples for _ in range(len(sample.features))])
    file_names = [sample.filename for sample in samples for _ in range(len(sample.features))]
    label_counts = Counter(sample.true_label for sample in samples)
    weights = []
    for sample in samples:
        # Each file has equal total mass inside its class; each class gets equal mass.
        per_window = 1.0 / max(label_counts[sample.true_label], 1) / max(len(sample.features), 1)
        weights.extend([per_window] * len(sample.features))
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(float(weights.mean()), 1e-12)
    return x, y, weights, file_names


def fit_model(samples: list[LateralSample], method: str) -> LateralFineModel:
    x, y, weights, file_names = build_training_matrix(samples)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    prototypes: dict[str, list[np.ndarray]] = defaultdict(list)
    sources: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        prototypes[sample.true_label].append(sample.features.mean(axis=0))
        sources[sample.true_label].append(sample.filename)
    for label in sorted(set(y)):
        prototypes[label].append(x[y == label].mean(axis=0))
        sources[label].append(f"{label}::class_center")
    base_model = LateralFineModel(
        prototypes={label: list(vectors) for label, vectors in prototypes.items()},
        prototype_sources={label: list(items) for label, items in sources.items()},
        feature_mean=mean,
        feature_scale=scale,
        margin_threshold=0.10 if method != "physical_hybrid" else 0.12,
        confidence_threshold=0.50,
        distance_thresholds={},
        pair_margin_thresholds={},
        classifier=None,
        feature_names=list(LATERAL_FEATURE_NAMES),
    )
    same_distances: dict[str, list[float]] = defaultdict(list)
    pair_margins: dict[str, list[float]] = defaultdict(list)
    for features, label in zip(x, y):
        pred = base_model.predict_from_features(features)
        if pred["lateral_prototype_label"] == label:
            same_distances[label].append(float(pred["lateral_prototype_distance"]))
        pair_key = "::".join(sorted([str(label), str(pred["lateral_second_label"])]))
        pair_margins[pair_key].append(float(pred["lateral_prototype_margin"]))
    distance_thresholds = {
        label: round(float(np.percentile(values, 92) * 1.20), 6) if values else 1.0
        for label, values in same_distances.items()
    }
    pair_thresholds = {
        key: round(max(0.08, float(np.percentile(values, 18)) * 0.75), 6)
        for key, values in pair_margins.items()
        if values
    }
    classifier = build_classifier(method)
    if classifier is not None:
        scaled = (x - mean) / scale
        try:
            classifier.fit(scaled, y, sample_weight=weights)
        except TypeError:
            classifier.fit(scaled, y)
    return LateralFineModel(
        prototypes={label: list(vectors) for label, vectors in prototypes.items()},
        prototype_sources={label: list(items) for label, items in sources.items()},
        feature_mean=mean,
        feature_scale=scale,
        margin_threshold=0.10 if method != "physical_hybrid" else 0.12,
        confidence_threshold=0.50,
        distance_thresholds=distance_thresholds,
        pair_margin_thresholds=pair_thresholds,
        classifier=classifier,
        feature_names=list(LATERAL_FEATURE_NAMES),
        submodel_version="lateral_subclassifier_v2_3_candidate",
    )


def build_classifier(method: str) -> object | None:
    if method in {"prototype", "physical_hybrid"}:
        return None
    if method == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    if method == "linear_svm":
        from sklearn.svm import LinearSVC

        return LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return LinearDiscriminantAnalysis()
    if method == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=60, random_state=42, min_samples_leaf=2, class_weight="balanced")
    raise ValueError(method)


def predict_file(model: LateralFineModel, sample: LateralSample, method: str) -> dict[str, Any]:
    rows = [model.predict_from_features(features) for features in sample.features]
    labels = [row["lateral_posture_label"] for row in rows]
    accepted = [label for label in labels if label != LATERAL_BOUNDARY_LABEL]
    counts = Counter(accepted)
    if counts:
        final, top_count = counts.most_common(1)[0]
        stable_ratio = top_count / max(len(rows), 1)
    else:
        final, stable_ratio = LATERAL_BOUNDARY_LABEL, 0.0
    boundary_ratio = labels.count(LATERAL_BOUNDARY_LABEL) / max(len(rows), 1)
    # File-level safety: if a file is mostly boundary or the accepted label is weak,
    # report the coarse lateral fallback instead of forcing a three-way decision.
    if final != LATERAL_BOUNDARY_LABEL and (stable_ratio < 0.45 or (boundary_ratio > 0.68 and stable_ratio < 0.55)):
        final = LATERAL_BOUNDARY_LABEL
    wrong_accept = final not in {sample.true_label, LATERAL_BOUNDARY_LABEL}
    correct_accept = final == sample.true_label
    correct_fallback = final == LATERAL_BOUNDARY_LABEL
    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    return {
        "filename": sample.filename,
        "batch": sample.batch,
        "true_label": sample.true_label,
        "method": method,
        "final_lateral_label": final,
        "file_result_type": "correct_accept" if correct_accept else "correct_fallback" if correct_fallback else "wrong_accept",
        "correct_accept": correct_accept,
        "correct_fallback": correct_fallback,
        "wrong_accept": wrong_accept,
        "lateral_boundary_ratio": round(float(boundary_ratio), 6),
        "lateral_stable_ratio": round(float(stable_ratio), 6),
        "lateral_switch_count": switches,
        "mean_lateral_confidence": round(float(np.mean([row["lateral_confidence"] for row in rows])), 6),
        "mean_lateral_margin": round(float(np.mean([row["lateral_margin"] for row in rows])), 6),
        "prototype_label_mode": Counter(row["lateral_prototype_label"] for row in rows).most_common(1)[0][0],
        "nearest_prototype_distance_mean": round(float(np.mean([row["lateral_prototype_distance"] for row in rows])), 6),
        "window_count": len(rows),
        "accepted_label_counts": json.dumps(dict(counts), ensure_ascii=False),
        "boundary_reasons": "; ".join(sorted({reason for row in rows for reason in row["lateral_boundary_reasons"]})),
    }


def run_lofo(samples: list[LateralSample], method: str) -> list[dict[str, Any]]:
    rows = []
    for test in samples:
        train = [sample for sample in samples if sample.filename != test.filename]
        model = fit_model(train, method)
        row = predict_file(model, test, method)
        row["fold"] = f"leave_out:{test.filename}"
        row["train_files"] = ";".join(sample.filename for sample in train)
        rows.append(row)
    return rows


def batch_side_transfer(samples: list[LateralSample], method: str, train_side_batch: str, test_side_batch: str) -> list[dict[str, Any]]:
    train = [s for s in samples if s.true_label != SIDE_LEANING_LABEL or s.batch == train_side_batch]
    test = [s for s in samples if s.true_label == SIDE_LEANING_LABEL and s.batch == test_side_batch]
    model = fit_model(train, method)
    rows = []
    for sample in test:
        row = predict_file(model, sample, method)
        row["fold"] = f"train_{train_side_batch}_test_{test_side_batch}"
        row["train_files"] = ";".join(item.filename for item in train)
        rows.append(row)
    return rows


def summarize_method(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("correct_accept")))
    fallback = sum(1 for row in rows if bool(row.get("correct_fallback")))
    wrong = sum(1 for row in rows if bool(row.get("wrong_accept")))
    return {
        "method": method,
        "file_count": total,
        "correct_accept_count": correct,
        "correct_fallback_count": fallback,
        "wrong_accept_count": wrong,
        "fine_file_accuracy": round(correct / max(total, 1), 6),
        "safe_resolution_rate": round((correct + fallback) / max(total, 1), 6),
        "fallback_rate": round(fallback / max(total, 1), 6),
        "wrong_accept_rate": round(wrong / max(total, 1), 6),
        "mean_boundary_ratio": round(float(np.mean([float(row["lateral_boundary_ratio"]) for row in rows])) if rows else 0.0, 6),
    }


def choose_method(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["wrong_accept_count"]),
            -float(row["safe_resolution_rate"]),
            -int(row["correct_accept_count"]),
            int(row["correct_fallback_count"]),
            0 if row["method"] in {"prototype", "physical_hybrid"} else 1,
        ),
    )
    return str(ordered[0]["method"])


def split_rows(samples: list[LateralSample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        rows.append({
            "fold": f"leave_out:{sample.filename}",
            "test_file": sample.filename,
            "train_files": ";".join(item.filename for item in samples if item.filename != sample.filename),
            "grouping": "Leave-One-Independent-CSV-Out",
        })
    rows.append({"fold": "SY1_to_SY2", "test_file": "batch:SY2 side leaning", "train_files": "all CS1/XC1 + SY1 side leaning", "grouping": "Batch transfer"})
    rows.append({"fold": "SY2_to_SY1", "test_file": "batch:SY1 side leaning", "train_files": "all CS1/XC1 + SY2 side leaning", "grouping": "Batch transfer"})
    return rows


def lateral_gate_analysis(samples: list[LateralSample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        data = load_csv_playback(sample.path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_2_candidate"))
        session.process_all()
        posture_records = [record for record in session.predictions if record.display_status == "POSTURE"]
        raw_labels = Counter(record.raw_label for record in posture_records if record.raw_label)
        second_labels = Counter(record.second_label for record in posture_records if record.second_label)
        parent_labels = Counter(record.parent_posture_label for record in posture_records if record.parent_posture_label)
        pseudo = {
            "label": session.summary().get("main_posture") or (raw_labels.most_common(1)[0][0] if raw_labels else ""),
            "raw_label": raw_labels.most_common(1)[0][0] if raw_labels else "",
            "second_label": second_labels.most_common(1)[0][0] if second_labels else "",
            "parent_posture_label": parent_labels.most_common(1)[0][0] if parent_labels else "",
            "prototype_diagnosis": None,
            "subclassifier_triggered": any(bool(record.subclassifier_triggered) for record in posture_records),
            "is_boundary": any(bool(record.is_boundary) for record in posture_records),
        }
        fmap = sample.feature_maps[len(sample.feature_maps) // 2]
        triggered, reasons = should_run_lateral_subclassifier(pseudo, fmap)
        rows.append({
            "filename": sample.filename,
            "true_label": sample.true_label,
            "v2_2_main_posture": session.summary().get("main_posture"),
            "v2_2_raw_mode": pseudo["raw_label"],
            "v2_2_second_mode": pseudo["second_label"],
            "v2_2_leanback_triggered": pseudo["subclassifier_triggered"],
            "lateral_gate_triggered": triggered,
            "lateral_gate_reasons": "; ".join(reasons),
        })
    return rows


def boundary_case_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        for row in group:
            if row["filename"] in BOUNDARY_FILES:
                item = dict(row)
                item["boundary_case"] = BOUNDARY_FILES[row["filename"]]
                item["preferred_behavior"] = "correct fine if clear, otherwise 侧向坐姿 safe fallback"
                rows.append(item)
    return rows


def multiprototype_rows(model: LateralFineModel) -> list[dict[str, Any]]:
    rows = []
    for label, protos in model.prototypes.items():
        sources = model.prototype_sources.get(label, [])
        for idx, vector in enumerate(protos):
            rows.append({
                "prototype_label": label,
                "prototype_source": sources[idx] if idx < len(sources) else f"prototype_{idx}",
                "feature_count": len(vector),
                "cop_x": round(float(vector[LATERAL_FEATURE_NAMES.index("cop_x")]), 6),
                "cop_y": round(float(vector[LATERAL_FEATURE_NAMES.index("cop_y")]), 6),
                "left_share": round(float(vector[LATERAL_FEATURE_NAMES.index("left_share")]), 6),
                "front_share": round(float(vector[LATERAL_FEATURE_NAMES.index("front_share")]), 6),
            })
    return rows


def feature_rows(model: LateralFineModel, samples: list[LateralSample]) -> list[dict[str, Any]]:
    x, y, _, _ = build_training_matrix(samples)
    rows = []
    labels = [STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL]
    for index, name in enumerate(LATERAL_FEATURE_NAMES):
        row = {"feature": name}
        for label in labels:
            row[f"{label}_mean"] = round(float(x[y == label, index].mean()), 6)
        row["max_minus_min"] = round(max(row[f"{label}_mean"] for label in labels) - min(row[f"{label}_mean"] for label in labels), 6)
        rows.append(row)
    return rows


def threshold_rows(model: LateralFineModel) -> list[dict[str, Any]]:
    rows = [
        {"threshold_type": "lateral_margin", "label": "all", "value": model.margin_threshold},
        {"threshold_type": "lateral_confidence", "label": "all", "value": model.confidence_threshold},
    ]
    for label, value in (model.distance_thresholds or {}).items():
        rows.append({"threshold_type": "prototype_distance", "label": label, "value": value})
    for label, value in (model.pair_margin_thresholds or {}).items():
        rows.append({"threshold_type": "pair_margin", "label": label, "value": value})
    return rows


def save_artifacts(
    samples: list[LateralSample],
    model: LateralFineModel,
    selected_method: str,
    comparison_rows: list[dict[str, Any]],
    lofo_rows: list[dict[str, Any]],
    sy1_sy2_rows: list[dict[str, Any]],
    sy2_sy1_rows: list[dict[str, Any]],
) -> None:
    save_lateral_fine_model(SUBMODEL_PATH, model)
    save_lateral_prototype_bank(SUBMODEL_PROTOTYPE_PATH, model)
    runtime = {
        "model_version": "v2_3_candidate",
        "display_name": "V2.3候选（侧向三类局部解析，未闭卷）",
        "parent_model_version": "v2_2_candidate",
        "lateral_submodel_version": model.submodel_version,
        "lateral_gate_rules": [
            "parent/raw/second/prototype candidate in 标准侧坐/斜跨坐/侧身倚靠坐",
            "parent boundary near a lateral candidate",
            "seat-observable physical single-side loading gate",
            "leanback two-stage gate has priority when both match",
        ],
        "fallback_label": LATERAL_BOUNDARY_LABEL,
        "boundary_rules": [
            "low_classifier_margin",
            "low_prototype_margin",
            "classifier_prototype_conflict",
            "side_leaning_diagonal_overlap",
            "side_leaning_standard_side_overlap",
            "out_of_distribution",
            "gate_conflict",
        ],
        "margin_threshold": model.margin_threshold,
        "confidence_threshold": model.confidence_threshold,
        "distance_thresholds": model.distance_thresholds or {},
        "pair_margin_thresholds": model.pair_margin_thresholds or {},
        "default_model_pointer_changed": False,
    }
    write_json(SUBMODEL_RUNTIME_CONFIG_PATH, runtime)
    metadata = {
        "model_version": "v2_3_candidate",
        "model_name": "lateral_subclassifier_v2_3_candidate",
        "candidate_name": "V2.3 lateral three-class local resolver",
        "production_status": "candidate_only",
        "final_holdout_status": "not_started",
        "parent_model_version": "v2_2_candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": [STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL],
        "fallback_label": LATERAL_BOUNDARY_LABEL,
        "selected_method": selected_method,
        "feature_names": LATERAL_FEATURE_NAMES,
        "training_files": [sample.filename for sample in samples],
        "training_file_hashes": {sample.filename: sha256_file(sample.path) for sample in samples},
        "development_data_notice": "SY1/SY2/XC1/CS1 were used for design and validation; not eligible for future final holdout.",
        "classifier_type": selected_method,
        "prototype_strategy": "per-file prototypes plus class centers; validation folds rebuild prototypes without the held-out CSV",
        "cross_validation_scheme": ["Leave-One-Independent-CSV-Out", "SY1->SY2 side-leaning batch transfer", "SY2->SY1 side-leaning batch transfer"],
        "comparison_summary": comparison_rows,
        "lofo_summary": summarize_method(selected_method, lofo_rows),
        "sy1_train_sy2_summary": summarize_method("SY1->SY2", sy1_sy2_rows),
        "sy2_train_sy1_summary": summarize_method("SY2->SY1", sy2_sy1_rows),
        "python": {"version": sys.version, "platform": platform.platform()},
        "dependencies": dependency_versions(),
        "git_commit": git_commit(),
    }
    write_json(SUBMODEL_METADATA_PATH, metadata)
    bundle = {
        "model_version": "v2_3_candidate",
        "display_name": "V2.3候选（侧向三类局部解析，未闭卷）",
        "parent_model_version": "v2_2_candidate",
        "parent_bundle": str(MODEL_DIR / "v2_2_candidate.model_bundle.json"),
        "parent_rf_model": str(MODEL_DIR / "rf_posture_v2_1_candidate.joblib"),
        "parent_metadata": str(MODEL_DIR / "rf_posture_v2_1_candidate.metadata.json"),
        "parent_prototype_bank": str(MODEL_DIR / "prototype_bank_v2_1_candidate.json"),
        "parent_runtime_config": str(MODEL_DIR / "rf_posture_v2_1_candidate.runtime_config.json"),
        "leanback_submodel": str(MODEL_DIR / "leanback_subclassifier_v2_2_candidate.joblib"),
        "leanback_prototype_bank": str(MODEL_DIR / "leanback_prototype_bank_v2_2_candidate.json"),
        "leanback_runtime_config": str(MODEL_DIR / "leanback_subclassifier_v2_2_candidate.runtime_config.json"),
        "lateral_submodel": str(SUBMODEL_PATH),
        "lateral_metadata": str(SUBMODEL_METADATA_PATH),
        "lateral_prototype_bank": str(SUBMODEL_PROTOTYPE_PATH),
        "lateral_runtime_config": str(SUBMODEL_RUNTIME_CONFIG_PATH),
        "default_model_pointer_changed": False,
    }
    write_json(MODEL_BUNDLE_PATH, bundle)


def lateral_gate_regression(samples: list[LateralSample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        data = load_csv_playback(sample.path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_3_candidate"))
        session.process_all()
        posture_records = [record for record in session.predictions if record.display_status == "POSTURE"]
        triggers = [record for record in posture_records if record.lateral_subclassifier_triggered]
        labels = Counter(record.posture for record in posture_records if record.posture)
        rows.append({
            "filename": sample.filename,
            "true_label": sample.true_label,
            "frame_count": data.frame_count,
            "processed_frames": len(session.predictions),
            "export_complete": len(session.predictions) == data.frame_count,
            "lateral_trigger_count": len(triggers),
            "lateral_trigger_rate": round(len(triggers) / max(len(posture_records), 1), 6),
            "main_posture": session.summary().get("main_posture"),
            "posture_counts": json.dumps(dict(labels), ensure_ascii=False),
            "model_version": session.summary().get("model_version"),
        })
    return rows


def non_lateral_false_trigger_report() -> list[dict[str, Any]]:
    rows = []
    holdout_dir = PROJECT_ROOT / "posture_dataset_v2" / "external_holdout" / "holdout_batch_02"
    for path in sorted(holdout_dir.glob("*.csv")):
        if "manifest" in path.name:
            continue
        data = load_csv_playback(path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_3_candidate"))
        session.process_all()
        triggers = [record for record in session.predictions if record.lateral_subclassifier_triggered]
        rows.append({
            "filename": path.name,
            "frame_count": data.frame_count,
            "lateral_trigger_count": len(triggers),
            "lateral_trigger_rate": round(len(triggers) / max(data.frame_count, 1), 6),
            "main_posture": session.summary().get("main_posture"),
        })
    return rows


def leanback_regression_report() -> list[dict[str, Any]]:
    rows = []
    h3_dir = PROJECT_ROOT / "posture_dataset_v2" / "external_holdout" / "v2_2_h3_raw"
    for path in sorted(h3_dir.glob("H3_*.csv")):
        data = load_csv_playback(path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_3_candidate"))
        session.process_all()
        lateral_triggers = [record for record in session.predictions if record.lateral_subclassifier_triggered]
        leanback_triggers = [record for record in session.predictions if record.subclassifier_triggered]
        rows.append({
            "filename": path.name,
            "frame_count": data.frame_count,
            "main_posture": session.summary().get("main_posture"),
            "leanback_trigger_count": len(leanback_triggers),
            "lateral_trigger_count": len(lateral_triggers),
            "h3_houyangkaobei2_safe_fallback": path.name == "H3_houyangkaobei2.csv" and session.summary().get("main_posture") == "后靠坐姿",
        })
    return rows


def object_empty_unknown_gate_report() -> list[dict[str, Any]]:
    rows = []
    object_dir = PROJECT_ROOT / "recognizer" / "object_data" / "batch1_raw"
    for path in sorted(object_dir.glob("object_*.csv")):
        if "manifest" in path.name:
            continue
        data = load_csv_playback(path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_3_candidate"))
        session.process_all()
        posture_calls = sum(1 for record in session.predictions if record.occupancy_state != "HUMAN" and record.posture)
        lateral_triggers = sum(1 for record in session.predictions if record.lateral_subclassifier_triggered)
        rows.append({
            "filename": path.name,
            "frame_count": data.frame_count,
            "posture_call_count": posture_calls,
            "lateral_trigger_count": lateral_triggers,
            "main_status": session.summary().get("main_posture") or "no_posture",
        })
    api = Recognizer(model_version="v2_3_candidate")
    for name, frame in [("__empty_frame__", np.zeros((16, 16), dtype=float)), ("__unknown_low_load__", np.ones((16, 16), dtype=float) * 0.2)]:
        result = api.predict(frame)
        rows.append({
            "filename": name,
            "frame_count": 1,
            "posture_call_count": int(result.get("posture") is not None),
            "lateral_trigger_count": int(bool(result.get("lateral_subclassifier_triggered"))),
            "main_status": result.get("occupancy"),
        })
    return rows


def artifact_hashes() -> dict[str, Any]:
    paths = {
        "default_model": MODEL_DIR / "default_model.json",
        "v2_1_parent_model": MODEL_DIR / "rf_posture_v2_1_candidate.joblib",
        "v2_1_parent_metadata": MODEL_DIR / "rf_posture_v2_1_candidate.metadata.json",
        "v2_1_parent_prototype_bank": MODEL_DIR / "prototype_bank_v2_1_candidate.json",
        "v2_1_parent_runtime_config": MODEL_DIR / "rf_posture_v2_1_candidate.runtime_config.json",
        "v2_2_submodel": MODEL_DIR / "leanback_subclassifier_v2_2_candidate.joblib",
        "v2_2_prototype_bank": MODEL_DIR / "leanback_prototype_bank_v2_2_candidate.json",
        "v2_2_runtime_config": MODEL_DIR / "leanback_subclassifier_v2_2_candidate.runtime_config.json",
        "v2_2_bundle": MODEL_DIR / "v2_2_candidate.model_bundle.json",
        "v2_3_submodel": SUBMODEL_PATH,
        "v2_3_metadata": SUBMODEL_METADATA_PATH,
        "v2_3_prototype_bank": SUBMODEL_PROTOTYPE_PATH,
        "v2_3_runtime_config": SUBMODEL_RUNTIME_CONFIG_PATH,
        "v2_3_bundle": MODEL_BUNDLE_PATH,
    }
    payload = {key: {"path": str(path), "sha256": sha256_file(path), "exists": path.exists()} for key, path in paths.items()}
    payload["default_model_version"] = default_model_version()
    return payload


def v22_hashes(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in payload if key.startswith("v2_1") or key.startswith("v2_2") or key == "default_model" or key == "default_model_version"}


def artifact_manifest(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_version": "v2_3_candidate",
        "production_status": "candidate_only",
        "default_model_unchanged": before.get("default_model") == after.get("default_model"),
        "default_model_version": default_model_version(),
        "v2_2_artifacts_unchanged": v22_hashes(before) == v22_hashes(after),
        "before": before,
        "after": after,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }


def generate_plots(samples: list[LateralSample], rows: list[dict[str, Any]], model: LateralFineModel) -> None:
    _bar_plot(
        PLOT_DIR / "lofo_results.png",
        [row["filename"].replace(".csv", "") for row in rows],
        [1.0 if row["correct_accept"] else 0.0 for row in rows],
        "LOFO correct_accept",
    )
    x, y, _, _ = build_training_matrix(samples)
    _scatter_plot(
        PLOT_DIR / "boundary_distribution.png",
        x[:, LATERAL_FEATURE_NAMES.index("left_right_balance")],
        x[:, LATERAL_FEATURE_NAMES.index("front_back_balance")],
        list(y),
        "LR balance vs FB balance",
    )
    distances = []
    labels_for_dist = []
    for sample in samples:
        for features in sample.features:
            pred = model.predict_from_features(features)
            distances.append(float(pred["lateral_prototype_distance"]))
            labels_for_dist.append(sample.true_label)
    _scatter_plot(
        PLOT_DIR / "prototype_distance_distribution.png",
        np.arange(len(distances), dtype=float),
        np.asarray(distances, dtype=float),
        labels_for_dist,
        "Prototype distances",
    )
    _confusion_plot(PLOT_DIR / "confusion_matrix.png", rows)


def _label_color(label: str) -> tuple[int, int, int]:
    return {
        STANDARD_SIDE_SITTING_LABEL: (47, 128, 237),
        DIAGONAL_SITTING_LABEL: (242, 153, 74),
        SIDE_LEANING_LABEL: (39, 174, 96),
        LATERAL_BOUNDARY_LABEL: (155, 89, 182),
    }.get(label, (120, 120, 120))


def _bar_plot(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 1100, 360
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), title, fill=(20, 20, 20))
    margin_left, margin_bottom = 50, 70
    chart_w, chart_h = width - 80, height - 120
    draw.rectangle((margin_left, 40, margin_left + chart_w, 40 + chart_h), outline=(210, 210, 210))
    bar_w = max(8, chart_w // max(len(values), 1) - 6)
    for i, value in enumerate(values):
        x0 = margin_left + i * (chart_w / max(len(values), 1)) + 3
        bar_h = int(chart_h * max(0.0, min(1.0, value)))
        y0 = 40 + chart_h - bar_h
        color = (47, 128, 237) if value >= 0.99 else (242, 153, 74)
        draw.rectangle((int(x0), y0, int(x0 + bar_w), 40 + chart_h), fill=color)
        draw.text((int(x0), 40 + chart_h + 8), labels[i][:14], fill=(60, 60, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _scatter_plot(path: Path, xs: np.ndarray, ys: np.ndarray, labels: list[str], title: str) -> None:
    width, height = 760, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), title, fill=(20, 20, 20))
    x0, y0, w, h = 60, 50, width - 110, height - 110
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(210, 210, 210))
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if abs(xmax - xmin) < 1e-9:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0
    for x, y, label in zip(xs, ys, labels):
        px = int(x0 + (float(x) - xmin) / (xmax - xmin) * w)
        py = int(y0 + h - (float(y) - ymin) / (ymax - ymin) * h)
        color = _label_color(label)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    legend_y = y0 + h + 20
    for i, label in enumerate([STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL, LATERAL_BOUNDARY_LABEL]):
        lx = x0 + i * 160
        draw.rectangle((lx, legend_y, lx + 12, legend_y + 12), fill=_label_color(label))
        draw.text((lx + 16, legend_y - 2), label, fill=(50, 50, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _confusion_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    truth_labels = [STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL]
    pred_labels = [STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL, LATERAL_BOUNDARY_LABEL]
    matrix = {truth: Counter() for truth in truth_labels}
    for row in rows:
        matrix[row["true_label"]][row["final_lateral_label"]] += 1
    cell = 92
    width, height = 560, 410
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), "LOFO confusion matrix", fill=(20, 20, 20))
    ox, oy = 120, 70
    for c, pred in enumerate(pred_labels):
        draw.text((ox + c * cell, oy - 24), pred[:5], fill=(40, 40, 40))
    for r, truth in enumerate(truth_labels):
        draw.text((16, oy + r * cell + 32), truth, fill=(40, 40, 40))
        for c, pred in enumerate(pred_labels):
            val = matrix[truth].get(pred, 0)
            intensity = 245 - min(180, val * 60)
            draw.rectangle((ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 4, oy + (r + 1) * cell - 4), fill=(intensity, intensity, 255), outline=(210, 210, 210))
            draw.text((ox + c * cell + 38, oy + r * cell + 34), str(val), fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_report(
    samples: list[LateralSample],
    selected_method: str,
    comparison_rows: list[dict[str, Any]],
    lofo_rows: list[dict[str, Any]],
    sy1_sy2_rows: list[dict[str, Any]],
    sy2_sy1_rows: list[dict[str, Any]],
    non_lateral_rows: list[dict[str, Any]],
    leanback_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> None:
    lofo = summarize_method(selected_method, lofo_rows)
    sy1sy2 = summarize_method("SY1->SY2", sy1_sy2_rows)
    sy2sy1 = summarize_method("SY2->SY1", sy2_sy1_rows)
    non_lateral_triggers = sum(int(row["lateral_trigger_count"]) for row in non_lateral_rows)
    leanback_lateral_triggers = sum(int(row["lateral_trigger_count"]) for row in leanback_rows)
    object_triggers = sum(int(row["lateral_trigger_count"]) for row in object_rows)
    object_posture_calls = sum(int(row.get("posture_call_count", 0)) for row in object_rows)
    boundary_rows = boundary_case_rows(lofo_rows)
    lines = [
        "# V2.3 Lateral Local Resolver Candidate Report",
        "",
        "本报告只使用 SY1/SY2/XC1/CS1 development CSV；它们已经参与设计决策，不能作为未来最终闭卷数据。",
        "",
        f"- Parent model: `v2_2_candidate`",
        f"- Selected method: `{selected_method}`",
        f"- LOFO correct_accept: {lofo['correct_accept_count']}/{lofo['file_count']}",
        f"- LOFO correct_fallback: {lofo['correct_fallback_count']}/{lofo['file_count']}",
        f"- LOFO wrong_accept: {lofo['wrong_accept_count']}/{lofo['file_count']}",
        f"- SY1→SY2 correct_accept: {sy1sy2['correct_accept_count']}/{sy1sy2['file_count']}, wrong_accept: {sy1sy2['wrong_accept_count']}",
        f"- SY2→SY1 correct_accept: {sy2sy1['correct_accept_count']}/{sy2sy1['file_count']}, wrong_accept: {sy2sy1['wrong_accept_count']}",
        f"- Non-lateral false trigger count: {non_lateral_triggers}",
        f"- H3 leanback lateral trigger count: {leanback_lateral_triggers}",
        f"- Object/EMPTY/UNKNOWN lateral trigger count: {object_triggers}",
        f"- Object posture calls: {object_posture_calls}",
        f"- Default model remains: `{default_model_version()}`",
        f"- V2.2 artifacts unchanged: {artifacts['v2_2_artifacts_unchanged']}",
        "",
        "## Candidate Comparison",
        "",
        "| method | correct_accept | fallback | wrong_accept | safe_resolution | mean boundary |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['method']} | {row['correct_accept_count']}/{row['file_count']} | "
            f"{row['correct_fallback_count']} | {row['wrong_accept_count']} | {row['safe_resolution_rate']} | {row['mean_boundary_ratio']} |"
        )
    lines.extend([
        "",
        "## Boundary Files",
        "",
        "| filename | truth | final | result | boundary_ratio | reasons |",
        "|---|---|---|---|---:|---|",
    ])
    for row in boundary_rows:
        lines.append(
            f"| {row['filename']} | {row['true_label']} | {row['final_lateral_label']} | {row['file_result_type']} | "
            f"{row['lateral_boundary_ratio']} | {row['boundary_reasons']} |"
        )
    lines.extend([
        "",
        "## Saved Candidate Artifacts",
        "",
        f"- `{SUBMODEL_PATH}`",
        f"- `{SUBMODEL_METADATA_PATH}`",
        f"- `{SUBMODEL_PROTOTYPE_PATH}`",
        f"- `{SUBMODEL_RUNTIME_CONFIG_PATH}`",
        f"- `{MODEL_BUNDLE_PATH}`",
        "",
        "## Next Closed-Book Holdout",
        "",
        "候选冻结后再采 6 份全新 CSV：侧身倚靠坐2份、斜跨坐2份、标准侧坐2份。该批不得用于训练、Prototype、门控或阈值调整。",
    ])
    (REPORT_DIR / "v2_3_candidate_report.md").write_text("\n".join(lines), encoding="utf-8")


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def dependency_versions() -> dict[str, str]:
    versions = {"numpy": np.__version__}
    for module in ["sklearn", "joblib"]:
        try:
            imported = __import__(module)
            versions[module] = str(getattr(imported, "__version__", "unknown"))
        except Exception as exc:
            versions[module] = f"unavailable: {exc}"
    return versions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
