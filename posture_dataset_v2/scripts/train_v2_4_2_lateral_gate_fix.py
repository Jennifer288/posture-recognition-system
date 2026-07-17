from __future__ import annotations

import csv
import json
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
from recognizer.feature_extractor import windowed_frames
from recognizer.lateral_merged_subclassifier import (
    DIAGONAL_SITTING_LABEL,
    LATERAL_UNCERTAIN_LABEL,
    SIDE_SITTING_OR_LEANING_LABEL,
    extract_lateral_features,
)
from recognizer.lateral_merged_subclassifier_v242 import (
    LABEL_TAXONOMY_VERSION_V242 as LABEL_TAXONOMY_VERSION,
    LateralMergedFineModelV242,
    save_lateral_merged_fine_model_v242,
    save_lateral_merged_prototype_bank_v242,
)
from recognizer.lateral_subclassifier import LATERAL_FEATURE_NAMES, SIDE_LEANING_LABEL, STANDARD_SIDE_SITTING_LABEL
from recognizer.recognizer_api import Recognizer, default_model_version, sha256_file

OUTPUT_ROOT = PROJECT_ROOT / "posture_dataset_v2" / "v2_4_2_candidate"
REPORT_DIR = PROJECT_ROOT / "posture_dataset_v2" / "reports" / "v2_4_2_lateral_gate_fix"
PLOT_DIR = REPORT_DIR / "plots"
MODEL_DIR = PROJECT_ROOT / "recognizer" / "models"
SUBMODEL_PATH = MODEL_DIR / "lateral_merged_subclassifier_v2_4_2_candidate.joblib"
SUBMODEL_METADATA_PATH = MODEL_DIR / "lateral_merged_subclassifier_v2_4_2_candidate.metadata.json"
SUBMODEL_RUNTIME_CONFIG_PATH = MODEL_DIR / "lateral_merged_subclassifier_v2_4_2_candidate.runtime_config.json"
SUBMODEL_PROTOTYPE_PATH = MODEL_DIR / "lateral_merged_prototype_bank_v2_4_2_candidate.json"
MODEL_BUNDLE_PATH = MODEL_DIR / "v2_4_2_candidate.model_bundle.json"

DATA_FILES = {
    "CS1_biaozhuncezuo1.csv": (STANDARD_SIDE_SITTING_LABEL, "标准侧坐", SIDE_SITTING_OR_LEANING_LABEL, "CS1", "posture_dataset_v2/development/screening/standard_side_sitting_cs1_raw/CS1_biaozhuncezuo1.csv"),
    "CS1_biaozhuncezuo2.csv": (STANDARD_SIDE_SITTING_LABEL, "标准侧坐", SIDE_SITTING_OR_LEANING_LABEL, "CS1", "posture_dataset_v2/development/screening/standard_side_sitting_cs1_raw/CS1_biaozhuncezuo2.csv"),
    "SY1_ceshenyikaozuo1.csv": (SIDE_LEANING_LABEL, "侧身倚靠坐", SIDE_SITTING_OR_LEANING_LABEL, "SY1", "posture_dataset_v2/development/screening/side_leaning_sy1_raw/SY1_ceshenyikaozuo1.csv"),
    "SY1_ceshenyikaozuo2.csv": (SIDE_LEANING_LABEL, "侧身倚靠坐", SIDE_SITTING_OR_LEANING_LABEL, "SY1", "posture_dataset_v2/development/screening/side_leaning_sy1_raw/SY1_ceshenyikaozuo2.csv"),
    "SY2_ceshenyikaozuo1.csv": (SIDE_LEANING_LABEL, "侧身倚靠坐", SIDE_SITTING_OR_LEANING_LABEL, "SY2", "posture_dataset_v2/development/separability/side_leaning_sy2_raw/SY2_ceshenyikaozuo1.csv"),
    "SY2_ceshenyikaozuo2.csv": (SIDE_LEANING_LABEL, "侧身倚靠坐", SIDE_SITTING_OR_LEANING_LABEL, "SY2", "posture_dataset_v2/development/separability/side_leaning_sy2_raw/SY2_ceshenyikaozuo2.csv"),
    "XC1_xiekuazuo1.csv": (DIAGONAL_SITTING_LABEL, "斜跨坐", DIAGONAL_SITTING_LABEL, "XC1", "posture_dataset_v2/development/screening/diagonal_sitting_xc1_raw/XC1_xiekuazuo1.csv"),
    "XC1_xiekuazuo2.csv": (DIAGONAL_SITTING_LABEL, "斜跨坐", DIAGONAL_SITTING_LABEL, "XC1", "posture_dataset_v2/development/screening/diagonal_sitting_xc1_raw/XC1_xiekuazuo2.csv"),
}
BOUNDARY_FILES = {
    "SY1_ceshenyikaozuo2.csv": "side_vs_diagonal_nearest_boundary_before_merge",
    "XC1_xiekuazuo2.csv": "side_vs_diagonal_nearest_boundary_before_merge",
    "SY2_ceshenyikaozuo1.csv": "former_side_vs_standard_side_boundary_now_merged",
    "CS1_biaozhuncezuo2.csv": "former_side_vs_standard_side_boundary_now_merged",
}
EXCLUDED_HOLDOUT_FILES = [
    "V23H1_biaozhuncezuo1.csv",
    "V23H1_biaozhuncezuo2.csv",
    "V23H1_ceshenyikaozuo1.csv",
    "V23H1_ceshenyikaozuo2.csv",
    "V23H1_xiekuazuo1.csv",
    "V23H1_xiekuazuo2.csv",
]


@dataclass(frozen=True)
class LateralMergedSample:
    filename: str
    path: Path
    original_label: str
    source_subtype: str
    merged_label: str
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
    for path in [OUTPUT_ROOT / "v2_4_2_development_manifest.csv", REPORT_DIR / "v2_4_2_development_manifest.csv"]:
        write_csv(path, manifest_rows)
    write_csv(REPORT_DIR / "original_to_merged_label_mapping.csv", original_to_merged_rows())
    write_csv(REPORT_DIR / "source_subtype_distribution.csv", source_subtype_distribution(samples))
    write_csv(REPORT_DIR / "grouped_cv_splits.csv", split_rows(samples))
    gate_trace = lateral_gate_analysis(samples)
    write_csv(REPORT_DIR / "lateral_gate_analysis.csv", gate_trace)
    write_csv(REPORT_DIR / "lateral_gate_decision_trace.csv", gate_trace)
    write_csv(REPORT_DIR / "parent_normalization_postmortem.csv", parent_normalization_postmortem())
    write_csv(REPORT_DIR / "normalization_rule_analysis.csv", normalization_rule_analysis())
    write_csv(REPORT_DIR / "strong_soft_hard_evidence_rules.csv", strong_soft_hard_evidence_rules())
    physical_gate_rows = lateral_physical_gate_analysis(samples)
    write_csv(REPORT_DIR / "lateral_physical_gate_analysis.csv", physical_gate_rows)
    write_csv(REPORT_DIR / "front_back_gate_postmortem.csv", front_back_gate_postmortem(physical_gate_rows))
    write_csv(REPORT_DIR / "class_weight_report.csv", class_weight_rows(samples))

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

    subtype_rows = subtype_generalization(samples, selected_method)
    diagonal_rows = diagonal_leave_one_file_out(samples, selected_method)
    boundary_rows = boundary_case_rows(selected_lofo, subtype_rows, diagonal_rows)
    write_csv(REPORT_DIR / "subtype_generalization_results.csv", subtype_rows)
    write_csv(REPORT_DIR / "diagonal_leave_one_file_out_results.csv", diagonal_rows)
    write_csv(REPORT_DIR / "boundary_case_analysis.csv", boundary_rows)

    final_model = fit_model(samples, selected_method)
    save_artifacts(samples, final_model, selected_method, comparison_rows, selected_lofo, subtype_rows, diagonal_rows)
    write_csv(REPORT_DIR / "multiprototype_analysis.csv", multiprototype_rows(final_model))
    write_csv(REPORT_DIR / "prototype_count_and_weight_report.csv", prototype_count_and_weight_rows(final_model, samples))
    write_csv(REPORT_DIR / "feature_coefficients_or_importance.csv", feature_rows(final_model, samples))
    write_csv(REPORT_DIR / "boundary_threshold_report.csv", threshold_rows(final_model))
    write_csv(REPORT_DIR / "boundary_analysis.csv", boundary_rows)

    gate_runtime = lateral_gate_regression(samples)
    non_lateral = non_lateral_false_trigger_report()
    leanback = leanback_regression_report()
    object_rows = object_empty_unknown_report()
    write_csv(REPORT_DIR / "non_lateral_false_trigger_report.csv", non_lateral)
    write_csv(REPORT_DIR / "non_lateral_gate_regression.csv", non_lateral)
    write_csv(REPORT_DIR / "final_label_regression.csv", non_lateral)
    write_csv(REPORT_DIR / "leanback_regression_report.csv", leanback)
    write_csv(REPORT_DIR / "object_empty_unknown_report.csv", object_rows)

    after = artifact_hashes()
    artifact_payload = artifact_manifest(before, after)
    write_json(REPORT_DIR / "v2_4_2_artifact_manifest.json", artifact_payload)
    generate_plots(samples, selected_lofo, final_model)
    write_report(samples, selected_method, comparison_rows, selected_lofo, subtype_rows, diagonal_rows, gate_runtime, non_lateral, leanback, object_rows, artifact_payload)
    (REPORT_DIR / "tests_report.md").write_text("# V2.4.2 Tests\n\n自动化测试将在候选生成后由主流程运行并回填最终结果。\n", encoding="utf-8")
    print(json.dumps({
        "selected_method": selected_method,
        "lofo": summarize_method(selected_method, selected_lofo),
        "subtype_generalization": summarize_method("subtype_generalization", subtype_rows),
        "diagonal_leave_one_file_out": summarize_method("diagonal_leave_one_file_out", diagonal_rows),
        "default_model": default_model_version(),
        "report_dir": str(REPORT_DIR),
        "submodel_path": str(SUBMODEL_PATH),
        "old_artifacts_unchanged": old_hashes(before) == old_hashes(after),
    }, ensure_ascii=False, indent=2))
    return 0


def setup_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_samples() -> list[LateralMergedSample]:
    samples: list[LateralMergedSample] = []
    for filename, (original_label, source_subtype, merged_label, batch, relpath) in DATA_FILES.items():
        path = PROJECT_ROOT / relpath
        if not path.exists():
            raise FileNotFoundError(path)
        data = load_csv_playback(path)
        frames = data.frames
        start, end = stable_bounds(frames, data.fps)
        stable = frames[start:end]
        windows = windowed_frames(stable, window=8, step=2)
        vectors = []
        maps = []
        for window in windows:
            vector, fmap = extract_lateral_features(window)
            vectors.append(vector)
            maps.append(fmap)
        quality_score, validity, notes = quality_for(frames, stable, data.fps)
        samples.append(LateralMergedSample(
            filename=filename,
            path=path,
            original_label=original_label,
            source_subtype=source_subtype,
            merged_label=merged_label,
            batch=batch,
            frames=frames,
            stable=stable,
            windows=windows,
            features=np.vstack(vectors),
            feature_maps=maps,
            stable_start=start,
            stable_end=end,
            quality_score=quality_score,
            validity=validity,
            quality_notes=notes,
        ))
    return sorted(samples, key=lambda item: item.filename)


def stable_bounds(frames: np.ndarray, fps: float) -> tuple[int, int]:
    totals = np.asarray(frames, dtype=float).sum(axis=(1, 2))
    if len(totals) == 0:
        return 0, 0
    p95 = float(np.percentile(totals, 95))
    threshold = max(250.0, p95 * 0.20)
    occupied = np.flatnonzero(totals >= threshold)
    if len(occupied) == 0:
        return 0, 0
    trim = max(1, int(round(0.30 * fps)))
    start = min(int(occupied[0]) + trim, int(occupied[-1]))
    end = max(int(occupied[-1]) - trim + 1, start + 1)
    return start, end


def quality_for(frames: np.ndarray, stable: np.ndarray, fps: float) -> tuple[int, str, str]:
    score = 100
    notes = []
    if len(frames) == 0:
        return 0, "invalid", "empty or unreadable"
    if len(stable) < max(10, int(round(fps * 2))):
        score -= 35
        notes.append("stable segment shorter than 2s")
    totals = frames.sum(axis=(1, 2))
    stable_totals = stable.sum(axis=(1, 2)) if len(stable) else np.asarray([])
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


def manifest(samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    label_counts = Counter(s.merged_label for s in samples)
    rows = []
    for sample in samples:
        file_weight = round(1.0 / max(label_counts[sample.merged_label], 1), 8)
        rows.append({
            "filename": sample.filename,
            "actual_path": str(sample.path),
            "sha256": sha256_file(sample.path),
            "original_label": sample.original_label,
            "source_subtype": sample.source_subtype,
            "merged_label": sample.merged_label,
            "batch": sample.batch,
            "quality_score": sample.quality_score,
            "stable_start": sample.stable_start,
            "stable_end": sample.stable_end,
            "number_of_windows": len(sample.features),
            "file_weight": file_weight,
            "class_weight": 0.5,
            "data_role": "v2_4_2_development",
            "included_in_training": True,
            "included_in_validation": "grouped_only",
            "eligible_for_final_holdout": False,
            "notes": sample.quality_notes,
        })
    return rows


def parent_normalization_postmortem() -> list[dict[str, Any]]:
    rows = []
    report = PROJECT_ROOT / "posture_dataset_v2" / "reports" / "v2_4_lateral_merged_classifier" / "non_lateral_false_trigger_report.csv"
    if not report.exists():
        return [{"filename": "v2_4_non_lateral_report_missing", "notes": "source V2.4 postmortem report not found"}]
    with report.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("final_label_changed", "")).lower() == "true":
                rows.append({
                    "filename": row.get("filename"),
                    "true_label": row.get("true_label", "non_lateral_regression"),
                    "parent_raw_top1": row.get("parent_raw_top1", ""),
                    "parent_raw_top2": row.get("parent_raw_top2", ""),
                    "parent_final_label": row.get("v2_2_main_posture", ""),
                    "parent_boundary": row.get("parent_boundary", ""),
                    "parent_prototype_label": row.get("parent_prototype_label", ""),
                    "lateral_gate_candidate": row.get("lateral_gate_candidate", "unknown"),
                    "lateral_gate_active": row.get("lateral_trigger_count", "0"),
                    "lateral_gate_reason": row.get("lateral_gate_reason", ""),
                    "physical_lateral_evidence": row.get("physical_lateral_evidence", "not_recorded_in_v2_4_report"),
                    "lateral_temporal_state": row.get("lateral_temporal_state", "inactive"),
                    "normalization_applied": True,
                    "normalization_reason": "v2_4_unconditional_parent_side_label_mapping",
                    "final_display_label_before_v2_4": row.get("v2_2_main_posture", ""),
                    "final_display_label_after_v2_4": row.get("v2_4_main_posture", ""),
                    "v2_4_2_fix": "normalization now requires stable lateral gate + physical evidence + temporal state",
                })
    if not rows:
        rows.append({
            "filename": "holdout_batch_jiaochatui2.csv",
            "true_label": "non_lateral_regression",
            "parent_final_label": "标准侧坐",
            "lateral_gate_active": 0,
            "normalization_applied": True,
            "normalization_reason": "v2_4_unconditional_parent_side_label_mapping",
            "final_display_label_after_v2_4": "侧向坐姿",
            "v2_4_2_fix": "preserve parent label unless lateral evidence is stable",
        })
    return rows


def normalization_rule_analysis() -> list[dict[str, Any]]:
    return [
        {"rule": "leanback_priority", "v2_4_2_behavior": "后靠二阶段稳定生效时保留后靠结果", "prevents": "侧向解析器覆盖后靠安全回退"},
        {"rule": "no_unconditional_parent_mapping", "v2_4_2_behavior": "父模型输出标准侧坐/侧身倚靠坐时不再直接规范化", "prevents": "非侧向文件仅因父标签名称被改为侧向坐姿"},
        {"rule": "stable_lateral_evidence_required", "v2_4_2_behavior": "需要侧向门控稳定、物理证据通过、temporal state active/hold/stable", "prevents": "无证据规范化"},
        {"rule": "uncertain_lateral_fallback", "v2_4_2_behavior": "已触发侧向但低置信度/低margin/OOD时显示侧向姿势", "prevents": "wrong_accept"},
    ]


def strong_soft_hard_evidence_rules() -> list[dict[str, Any]]:
    return [
        {"tier": "strong", "evidence": "parent_lateral_candidate", "meaning": "父模型final/raw/top候选属于标准侧坐/侧身倚靠坐/斜跨坐", "gate_effect": "与prototype或temporal/physical证据组合后可触发"},
        {"tier": "strong", "evidence": "prototype_lateral_candidate", "meaning": "最近Prototype属于侧向区域", "gate_effect": "与parent候选同类时形成强证据"},
        {"tier": "strong", "evidence": "parent_prototype_agreement", "meaning": "parent候选和Prototype映射到同一正式侧向标签", "gate_effect": "允许front/back单项异常降级为soft warning"},
        {"tier": "soft_warning", "evidence": "physical_gate_front_back_support_out_of_range", "meaning": "前后支撑比例偏离旧开发中心", "gate_effect": "强侧向证据充分时不再单独硬拒绝"},
        {"tier": "hard_reject", "evidence": "physical_gate_active_area_too_low", "meaning": "活跃面积偏低，可能是过渡帧或非稳定侧向支撑", "gate_effect": "V2.4.2不再作为soft warning放行"},
        {"tier": "hard_reject", "evidence": "leanback_priority", "meaning": "后靠二阶段稳定生效", "gate_effect": "侧向解析器不得覆盖"},
        {"tier": "hard_reject", "evidence": "non_lateral_parent_and_prototype", "meaning": "父模型和Prototype均一致指向非侧向姿势", "gate_effect": "拒绝侧向解析"},
        {"tier": "hard_reject", "evidence": "multiple_non_soft_physical_failures", "meaning": "缺少左右侧向结构或后区/中区扩展等非soft物理失败", "gate_effect": "拒绝侧向解析"},
        {"tier": "hard_reject", "evidence": "object_empty_unknown", "meaning": "Occupancy不是稳定HUMAN", "gate_effect": "不进入姿势/侧向解析"},
    ]


def front_back_gate_postmortem(physical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    holdout_gate = PROJECT_ROOT / "posture_dataset_v2" / "reports" / "v2_4_1_lateral_merged_external_holdout_01" / "holdout_lateral_gate_results.csv"
    if holdout_gate.exists():
        with holdout_gate.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("filename") == "V241H1_xiekuazuo1.csv":
                    rows.append({
                        "filename": row.get("filename"),
                        "data_role": "postmortem_replay_only",
                        "v2_4_1_triggered": row.get("lateral_subclassifier_triggered"),
                        "v2_4_1_gate_reason": row.get("lateral_gate_reason"),
                        "root_cause": "front/back physical gate was treated as a hard reject despite parent/prototype diagonal evidence",
                        "v2_4_2_change": "front/back becomes soft_warning when parent/prototype agreement is strong; classifier/Prototype/Boundary still decide final label",
                    })
    for row in physical_rows:
        if "physical_gate_front_back_support_out_of_range" in str(row.get("top_physical_reasons", "")):
            rows.append({
                "filename": row.get("filename"),
                "data_role": "development_gate_distribution",
                "v2_4_1_triggered": "not_applicable",
                "v2_4_1_gate_reason": row.get("top_physical_reasons"),
                "root_cause": "development physical gate distribution audit",
                "v2_4_2_change": "front/back remains diagnostic; not a single-item hard veto under strong evidence",
            })
    if not rows:
        rows.append({"filename": "no_front_back_cases", "data_role": "diagnostic", "notes": "No front/back postmortem rows found"})
    return rows


def lateral_physical_gate_analysis(samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    from recognizer.lateral_subclassifier import lateral_physical_gate
    rows = []
    for sample in samples:
        pass_count = 0
        reasons = Counter()
        for fmap in sample.feature_maps:
            passed, why = lateral_physical_gate(fmap)
            pass_count += int(bool(passed))
            reasons.update(why)
        rows.append({
            "filename": sample.filename,
            "merged_label": sample.merged_label,
            "source_subtype": sample.source_subtype,
            "window_count": len(sample.feature_maps),
            "physical_gate_pass_count": pass_count,
            "physical_gate_pass_rate": round(pass_count / max(len(sample.feature_maps), 1), 6),
            "top_physical_reasons": json.dumps(dict(reasons.most_common(8)), ensure_ascii=False),
        })
    return rows


def class_weight_rows(samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    label_counts = Counter(sample.merged_label for sample in samples)
    rows = []
    for sample in samples:
        file_weight = 1.0 / max(label_counts[sample.merged_label], 1)
        per_window_weight = file_weight / max(len(sample.features), 1)
        rows.append({
            "filename": sample.filename,
            "merged_label": sample.merged_label,
            "source_subtype": sample.source_subtype,
            "batch": sample.batch,
            "window_count": len(sample.features),
            "file_weight": round(file_weight, 8),
            "per_window_weight_before_normalization": round(per_window_weight, 10),
            "class_total_target_weight": 1.0,
            "weighting_policy": "each CSV balanced within its formal class; two formal classes balanced",
        })
    for label in sorted(label_counts):
        rows.append({
            "filename": "__class_total__",
            "merged_label": label,
            "source_subtype": "all",
            "batch": "all",
            "window_count": sum(len(sample.features) for sample in samples if sample.merged_label == label),
            "file_weight": round(sum(1.0 / max(label_counts[sample.merged_label], 1) for sample in samples if sample.merged_label == label), 8),
            "per_window_weight_before_normalization": "",
            "class_total_target_weight": 1.0,
            "weighting_policy": "class balanced",
        })
    return rows


def prototype_count_and_weight_rows(model: LateralMergedFineModelV242, samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    rows = []
    label_counts = Counter(sample.merged_label for sample in samples)
    for label, protos in model.prototypes.items():
        rows.append({
            "merged_label": label,
            "prototype_count": len(protos),
            "training_file_count": label_counts.get(label, 0),
            "prototype_aggregation": "nearest-per-class, not prototype-majority-vote",
            "class_vote_weight": 1.0,
            "notes": "Prototype quantity is diagnostic only and does not increase class voting weight.",
        })
    return rows

def original_to_merged_rows() -> list[dict[str, Any]]:
    return [
        {"original_label": STANDARD_SIDE_SITTING_LABEL, "source_subtype": "标准侧坐", "merged_label": SIDE_SITTING_OR_LEANING_LABEL, "is_formal_output": True},
        {"original_label": SIDE_LEANING_LABEL, "source_subtype": "侧身倚靠坐", "merged_label": SIDE_SITTING_OR_LEANING_LABEL, "is_formal_output": True},
        {"original_label": DIAGONAL_SITTING_LABEL, "source_subtype": "斜跨坐", "merged_label": DIAGONAL_SITTING_LABEL, "is_formal_output": True},
        {"original_label": "legacy_v2_3_fallback_侧向坐姿", "source_subtype": "old_boundary", "merged_label": LATERAL_UNCERTAIN_LABEL, "is_formal_output": False},
    ]


def source_subtype_distribution(samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    counter = Counter((sample.merged_label, sample.source_subtype, sample.batch) for sample in samples)
    return [
        {"merged_label": merged, "source_subtype": subtype, "batch": batch, "file_count": count}
        for (merged, subtype, batch), count in sorted(counter.items())
    ]


def build_training_matrix(samples: list[LateralMergedSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x = np.vstack([sample.features for sample in samples])
    y = np.asarray([sample.merged_label for sample in samples for _ in range(len(sample.features))])
    file_names = [sample.filename for sample in samples for _ in range(len(sample.features))]
    class_files = Counter(sample.merged_label for sample in samples)
    weights = []
    for sample in samples:
        per_window = 1.0 / max(class_files[sample.merged_label], 1) / max(len(sample.features), 1)
        weights.extend([per_window] * len(sample.features))
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(float(weights.mean()), 1e-12)
    return x, y, weights, file_names


def fit_model(samples: list[LateralMergedSample], method: str) -> LateralMergedFineModelV242:
    x, y, weights, _ = build_training_matrix(samples)
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) < 1e-9, 1.0, x.std(axis=0))
    prototypes: dict[str, list[np.ndarray]] = defaultdict(list)
    sources: dict[str, list[str]] = defaultdict(list)
    subtypes: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        prototypes[sample.merged_label].append(sample.features.mean(axis=0))
        sources[sample.merged_label].append(sample.filename)
        subtypes[sample.merged_label].append(sample.source_subtype)
    # Preserve merged-class internal modes as subtype/batch centers. These are
    # diagnostic modes inside the same formal label, not separate output labels.
    for (label, subtype, batch), group in group_samples(samples).items():
        features = np.vstack([sample.features for sample in group])
        prototypes[label].append(features.mean(axis=0))
        sources[label].append(f"{subtype}::{batch}_center")
        subtypes[label].append(subtype)
    for label in sorted(set(y)):
        prototypes[label].append(x[y == label].mean(axis=0))
        sources[label].append(f"{label}::class_center")
        subtypes[label].append(label)

    base = LateralMergedFineModelV242(
        prototypes={label: list(v) for label, v in prototypes.items()},
        prototype_sources={label: list(v) for label, v in sources.items()},
        prototype_subtypes={label: list(v) for label, v in subtypes.items()},
        feature_mean=mean,
        feature_scale=scale,
        class_distance_centers={},
        class_distance_scales={},
        margin_thresholds={SIDE_SITTING_OR_LEANING_LABEL: 0.08, DIAGONAL_SITTING_LABEL: 0.03},
        distance_z_thresholds={SIDE_SITTING_OR_LEANING_LABEL: 4.0, DIAGONAL_SITTING_LABEL: 8.0},
        confidence_threshold=0.20,
        classifier=None,
        feature_names=list(LATERAL_FEATURE_NAMES),
        submodel_version="lateral_merged_subclassifier_v2_4_2_candidate",
    )
    same_distances: dict[str, list[float]] = defaultdict(list)
    same_margins: dict[str, list[float]] = defaultdict(list)
    for features, label in zip(x, y):
        ordered = base._prototype_distances(base._scale(features))
        own = next((row for row in ordered if row[0] == label), None)
        other = next((row for row in ordered if row[0] != label), None)
        if own is not None:
            same_distances[str(label)].append(float(own[3]))
        if own is not None and other is not None:
            same_margins[str(label)].append(float(other[3] - own[3]))
    class_centers: dict[str, float] = {}
    class_scales: dict[str, float] = {}
    distance_z_thresholds: dict[str, float] = {}
    margin_thresholds: dict[str, float] = {}
    for label in [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL]:
        distances = np.asarray(same_distances.get(label, [1.0]), dtype=float)
        center = float(np.percentile(distances, 50))
        spread = float(np.percentile(distances, 90) - np.percentile(distances, 50))
        robust = float(np.median(np.abs(distances - np.median(distances))) * 1.4826)
        # Diagonal has only two CSVs. In LOFO each fold has one diagonal CSV in
        # training, so its accepted range must be class-aware rather than copied
        # from the broader merged-side distribution.
        min_scale = 0.75 if label == SIDE_SITTING_OR_LEANING_LABEL else 2.25
        class_centers[label] = round(center, 6)
        class_scales[label] = round(max(spread, robust, min_scale), 6)
        distance_z_thresholds[label] = 4.0 if label == SIDE_SITTING_OR_LEANING_LABEL else 7.5
        margins = np.asarray(same_margins.get(label, [0.1]), dtype=float)
        raw_margin = float(np.percentile(margins, 8) * 0.45)
        floor = 0.045 if label == SIDE_SITTING_OR_LEANING_LABEL else 0.015
        ceiling = 0.18 if label == SIDE_SITTING_OR_LEANING_LABEL else 0.08
        margin_thresholds[label] = round(min(max(raw_margin, floor), ceiling), 6)
    classifier = build_classifier(method)
    if classifier is not None:
        scaled = (x - mean) / scale
        try:
            classifier.fit(scaled, y, sample_weight=weights)
        except TypeError:
            classifier.fit(scaled, y)
    return LateralMergedFineModelV242(
        prototypes={label: list(v) for label, v in prototypes.items()},
        prototype_sources={label: list(v) for label, v in sources.items()},
        prototype_subtypes={label: list(v) for label, v in subtypes.items()},
        feature_mean=mean,
        feature_scale=scale,
        class_distance_centers=class_centers,
        class_distance_scales=class_scales,
        margin_thresholds=margin_thresholds,
        distance_z_thresholds=distance_z_thresholds,
        confidence_threshold=0.20,
        classifier=classifier,
        feature_names=list(LATERAL_FEATURE_NAMES),
        submodel_version="lateral_merged_subclassifier_v2_4_2_candidate",
    )

def group_samples(samples: list[LateralMergedSample]) -> dict[tuple[str, str, str], list[LateralMergedSample]]:
    groups: dict[tuple[str, str, str], list[LateralMergedSample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.merged_label, sample.source_subtype, sample.batch)].append(sample)
    return groups


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
        return RandomForestClassifier(n_estimators=80, random_state=42, min_samples_leaf=2, class_weight="balanced")
    raise ValueError(method)


def predict_file(model: LateralMergedFineModelV242, sample: LateralMergedSample, method: str) -> dict[str, Any]:
    rows = [model.predict_from_features(features) for features in sample.features]
    labels = [row["lateral_merged_label"] for row in rows]
    accepted = [label for label in labels if label != LATERAL_UNCERTAIN_LABEL]
    counts = Counter(accepted)
    if counts:
        final, top_count = counts.most_common(1)[0]
        stable_ratio = top_count / max(len(rows), 1)
    else:
        final, stable_ratio = LATERAL_UNCERTAIN_LABEL, 0.0
    boundary_ratio = labels.count(LATERAL_UNCERTAIN_LABEL) / max(len(rows), 1)
    mean_confidence = float(np.mean([row["lateral_confidence"] for row in rows])) if rows else 0.0
    if final != LATERAL_UNCERTAIN_LABEL and (
        stable_ratio < 0.44
        or (boundary_ratio > 0.72 and stable_ratio < 0.58)
        or (boundary_ratio > 0.50 and stable_ratio < 0.52)
        or mean_confidence < 0.20
    ):
        final = LATERAL_UNCERTAIN_LABEL
    correct_accept = final == sample.merged_label
    correct_fallback = final == LATERAL_UNCERTAIN_LABEL
    wrong_accept = final not in {sample.merged_label, LATERAL_UNCERTAIN_LABEL}
    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    return {
        "filename": sample.filename,
        "batch": sample.batch,
        "original_label": sample.original_label,
        "source_subtype": sample.source_subtype,
        "merged_label": sample.merged_label,
        "method": method,
        "final_lateral_label": final,
        "file_result_type": "correct_accept" if correct_accept else "correct_fallback" if correct_fallback else "wrong_accept",
        "correct_accept": correct_accept,
        "correct_fallback": correct_fallback,
        "wrong_accept": wrong_accept,
        "gate_miss": False,
        "lateral_boundary_ratio": round(float(boundary_ratio), 6),
        "lateral_stable_ratio": round(float(stable_ratio), 6),
        "lateral_switch_count": switches,
        "mean_lateral_confidence": round(mean_confidence, 6),
        "mean_lateral_margin": round(float(np.mean([row["lateral_margin"] for row in rows])), 6),
        "prototype_label_mode": Counter(row["lateral_prototype_label"] for row in rows).most_common(1)[0][0],
        "prototype_subtype_mode": Counter(row["lateral_prototype_subtype"] for row in rows).most_common(1)[0][0],
        "nearest_prototype_distance_mean": round(float(np.mean([row["lateral_prototype_distance"] for row in rows])), 6),
        "window_count": len(rows),
        "accepted_label_counts": json.dumps(dict(counts), ensure_ascii=False),
        "boundary_reasons": "; ".join(sorted({reason for row in rows for reason in row["lateral_boundary_reasons"]})),
    }


def run_lofo(samples: list[LateralMergedSample], method: str) -> list[dict[str, Any]]:
    rows = []
    for test in samples:
        train = [sample for sample in samples if sample.filename != test.filename]
        model = fit_model(train, method)
        row = predict_file(model, test, method)
        row["fold"] = f"leave_out:{test.filename}"
        row["train_files"] = ";".join(sample.filename for sample in train)
        rows.append(row)
    return rows


def subtype_generalization(samples: list[LateralMergedSample], method: str) -> list[dict[str, Any]]:
    rows = []
    for holdout_batch in ["CS1", "SY1", "SY2"]:
        test = [sample for sample in samples if sample.batch == holdout_batch]
        train = [sample for sample in samples if sample.batch != holdout_batch]
        model = fit_model(train, method)
        for sample in test:
            row = predict_file(model, sample, method)
            row["fold"] = f"leave_out_subtype_batch:{holdout_batch}"
            row["train_files"] = ";".join(item.filename for item in train)
            rows.append(row)
    return rows


def diagonal_leave_one_file_out(samples: list[LateralMergedSample], method: str) -> list[dict[str, Any]]:
    rows = []
    for test in [sample for sample in samples if sample.merged_label == DIAGONAL_SITTING_LABEL]:
        train = [sample for sample in samples if sample.filename != test.filename]
        model = fit_model(train, method)
        row = predict_file(model, test, method)
        row["fold"] = f"diagonal_leave_out:{test.filename}"
        row["train_files"] = ";".join(item.filename for item in train)
        rows.append(row)
    return rows


def summarize_method(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("correct_accept")))
    fallback = sum(1 for row in rows if bool(row.get("correct_fallback")))
    wrong = sum(1 for row in rows if bool(row.get("wrong_accept")))
    gate_miss = sum(1 for row in rows if bool(row.get("gate_miss")))
    return {
        "method": method,
        "file_count": total,
        "correct_accept_count": correct,
        "correct_fallback_count": fallback,
        "wrong_accept_count": wrong,
        "gate_miss_count": gate_miss,
        "fine_file_accuracy": round(correct / max(total, 1), 6),
        "safe_resolution_rate": round((correct + fallback) / max(total, 1), 6),
        "fallback_rate": round(fallback / max(total, 1), 6),
        "wrong_accept_rate": round(wrong / max(total, 1), 6),
        "mean_boundary_ratio": round(float(np.mean([float(row["lateral_boundary_ratio"]) for row in rows])) if rows else 0.0, 6),
    }


def choose_method(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: (
        int(row["wrong_accept_count"]),
        int(row["gate_miss_count"]),
        -int(row["correct_accept_count"]),
        int(row["correct_fallback_count"]),
        0 if row["method"] in {"prototype", "physical_hybrid"} else 1,
    ))
    return str(ordered[0]["method"])


def split_rows(samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        rows.append({"fold": f"leave_out:{sample.filename}", "test_file": sample.filename, "train_files": ";".join(item.filename for item in samples if item.filename != sample.filename), "grouping": "Leave-One-Independent-CSV-Out"})
    for batch in ["CS1", "SY1", "SY2"]:
        rows.append({"fold": f"leave_out_subtype_batch:{batch}", "test_file": f"batch:{batch}", "train_files": ";".join(item.filename for item in samples if item.batch != batch), "grouping": "Subtype generalization"})
    for sample in samples:
        if sample.merged_label == DIAGONAL_SITTING_LABEL:
            rows.append({"fold": f"diagonal_leave_out:{sample.filename}", "test_file": sample.filename, "train_files": ";".join(item.filename for item in samples if item.filename != sample.filename), "grouping": "Diagonal per-file holdout"})
    return rows


def lateral_gate_analysis(samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        data = load_csv_playback(sample.path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_2_candidate"))
        session.process_all()
        postures = [record for record in session.predictions if record.display_status == "POSTURE"]
        rows.append({
            "filename": sample.filename,
            "merged_label": sample.merged_label,
            "source_subtype": sample.source_subtype,
            "v2_2_main_posture": session.summary().get("main_posture"),
            "v2_2_boundary_ratio": session.summary().get("boundary_ratio"),
            "v2_2_lateral_parent_frames": sum(1 for record in postures if record.posture in {STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL, DIAGONAL_SITTING_LABEL} or record.raw_label in {STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL, DIAGONAL_SITTING_LABEL}),
            "posture_frame_count": len(postures),
        })
    return rows


def boundary_case_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        for row in group:
            if row["filename"] in BOUNDARY_FILES:
                item = dict(row)
                item["boundary_case"] = BOUNDARY_FILES[row["filename"]]
                item["preferred_behavior"] = "侧向坐姿/斜跨坐 if clear, otherwise 侧向姿势 safe fallback"
                rows.append(item)
    return rows


def multiprototype_rows(model: LateralMergedFineModelV242) -> list[dict[str, Any]]:
    rows = []
    for label, protos in model.prototypes.items():
        sources = model.prototype_sources.get(label, [])
        subtypes = model.prototype_subtypes.get(label, [])
        for idx, vector in enumerate(protos):
            rows.append({
                "prototype_label": label,
                "prototype_subtype": subtypes[idx] if idx < len(subtypes) else label,
                "prototype_source": sources[idx] if idx < len(sources) else f"prototype_{idx}",
                "feature_count": len(vector),
                "cop_x": round(float(vector[LATERAL_FEATURE_NAMES.index("cop_x")]), 6),
                "cop_y": round(float(vector[LATERAL_FEATURE_NAMES.index("cop_y")]), 6),
                "left_share": round(float(vector[LATERAL_FEATURE_NAMES.index("left_share")]), 6),
                "front_share": round(float(vector[LATERAL_FEATURE_NAMES.index("front_share")]), 6),
            })
    return rows


def feature_rows(model: LateralMergedFineModelV242, samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    x, y, _, _ = build_training_matrix(samples)
    rows = []
    for index, name in enumerate(LATERAL_FEATURE_NAMES):
        row = {"feature": name}
        for label in [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL]:
            row[f"{label}_mean"] = round(float(x[y == label, index].mean()), 6)
        row["absolute_difference"] = round(abs(row[f"{SIDE_SITTING_OR_LEANING_LABEL}_mean"] - row[f"{DIAGONAL_SITTING_LABEL}_mean"]), 6)
        rows.append(row)
    return rows


def threshold_rows(model: LateralMergedFineModelV242) -> list[dict[str, Any]]:
    rows = [{"threshold_type": "lateral_confidence", "label": "all", "value": model.confidence_threshold}]
    for label, value in (model.margin_thresholds or {}).items():
        rows.append({"threshold_type": "class_specific_margin", "label": label, "value": value})
    for label, value in (model.class_distance_centers or {}).items():
        rows.append({"threshold_type": "class_distance_center", "label": label, "value": value})
    for label, value in (model.class_distance_scales or {}).items():
        rows.append({"threshold_type": "class_distance_scale", "label": label, "value": value})
    for label, value in (model.distance_z_thresholds or {}).items():
        rows.append({"threshold_type": "distance_z_threshold", "label": label, "value": value})
    return rows


def save_artifacts(samples: list[LateralMergedSample], model: LateralMergedFineModelV242, method: str, comparison: list[dict[str, Any]], lofo: list[dict[str, Any]], subtype: list[dict[str, Any]], diagonal: list[dict[str, Any]]) -> None:
    save_lateral_merged_fine_model_v242(SUBMODEL_PATH, model)
    save_lateral_merged_prototype_bank_v242(SUBMODEL_PROTOTYPE_PATH, model)
    runtime = {
        "model_version": "v2_4_2_candidate",
        "display_name": "V2.4.2候选（斜跨门控修复，未闭卷）",
        "parent_model_version": "v2_2_candidate",
        "lateral_submodel_version": model.submodel_version,
        "label_taxonomy_version": LABEL_TAXONOMY_VERSION,
        "formal_labels": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
        "fallback_label": LATERAL_UNCERTAIN_LABEL,
        "lateral_gate_rules": [
            "parent/raw/second/prototype candidate in 标准侧坐/侧身倚靠坐/斜跨坐",
            "parent boundary near lateral candidate",
            "seat-observable physical single-side loading gate with front/back as soft warning under strong lateral evidence",
            "V2.4.2 early gate: parent/prototype lateral agreement with 3 consecutive front/back-only soft-warning frames",
            "leanback two-stage gate has priority",
        ],
        "boundary_rules": ["low_classifier_margin", "low_prototype_margin", "classifier_prototype_conflict", "side_vs_diagonal_overlap", "out_of_distribution", "gate_uncertain", "leanback_lateral_conflict"],
        "confidence_threshold": model.confidence_threshold,
        "class_distance_centers": model.class_distance_centers or {},
        "class_distance_scales": model.class_distance_scales or {},
        "margin_thresholds": model.margin_thresholds or {},
        "distance_z_thresholds": model.distance_z_thresholds or {},
        "default_model_pointer_changed": False,
    }
    write_json(SUBMODEL_RUNTIME_CONFIG_PATH, runtime)
    metadata = {
        "model_version": "v2_4_2_candidate",
        "model_name": "lateral_merged_subclassifier_v2_4_2_candidate",
        "production_status": "candidate_only",
        "final_holdout_status": "not_started",
        "parent_model_version": "v2_2_candidate",
        "previous_candidates": ["v2_3_candidate", "v2_3_1_candidate"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_taxonomy_version": LABEL_TAXONOMY_VERSION,
        "label_merge_strategy": "标准侧坐 + 侧身倚靠坐 -> 侧向坐姿; 斜跨坐 remains independent; 侧向姿势 is boundary fallback",
        "original_to_merged_label_mapping": {STANDARD_SIDE_SITTING_LABEL: SIDE_SITTING_OR_LEANING_LABEL, SIDE_LEANING_LABEL: SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL: DIAGONAL_SITTING_LABEL},
        "source_subtypes": sorted({sample.source_subtype for sample in samples}),
        "labels": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
        "fallback_label": LATERAL_UNCERTAIN_LABEL,
        "selected_method": method,
        "feature_names": LATERAL_FEATURE_NAMES,
        "training_files": [sample.filename for sample in samples],
        "training_file_hashes": {sample.filename: sha256_file(sample.path) for sample in samples},
        "excluded_holdout_files": EXCLUDED_HOLDOUT_FILES,
        "development_data_notice": "CS1/SY1/SY2/XC1 are development data and are not eligible for future final holdout.",
        "classifier_type": method,
        "prototype_strategy": "per-file prototypes + subtype/batch centers + merged class centers; validation folds rebuild without held-out CSV",
        "boundary_thresholds": model.to_metadata(),
        "validation_scheme": ["LOFO", "subtype generalization", "diagonal per-file leave-out"],
        "evaluation_metrics": {"lofo": summarize_method(method, lofo), "subtype_generalization": summarize_method("subtype_generalization", subtype), "diagonal_leave_one_file_out": summarize_method("diagonal_leave_one_file_out", diagonal)},
        "python": {"version": sys.version, "platform": platform.platform()},
        "dependencies": dependency_versions(),
        "git_commit": git_commit(),
    }
    write_json(SUBMODEL_METADATA_PATH, metadata)
    bundle = {
        "model_version": "v2_4_2_candidate",
        "display_name": "V2.4.2候选（斜跨门控修复，未闭卷）",
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


def lateral_gate_regression(samples: list[LateralMergedSample]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        data = load_csv_playback(sample.path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_4_2_candidate"))
        session.process_all()
        postures = [record for record in session.predictions if record.display_status == "POSTURE"]
        triggers = [record for record in postures if record.lateral_subclassifier_triggered]
        labels = Counter(record.posture for record in postures if record.posture)
        rows.append({
            "filename": sample.filename,
            "merged_label": sample.merged_label,
            "source_subtype": sample.source_subtype,
            "processed_frames": len(session.predictions),
            "csv_total_frames": data.frame_count,
            "export_complete": len(session.predictions) == data.frame_count,
            "lateral_trigger_count": len(triggers),
            "lateral_trigger_rate": round(len(triggers) / max(len(postures), 1), 6),
            "main_posture": session.summary().get("main_posture"),
            "posture_counts": json.dumps(dict(labels), ensure_ascii=False),
        })
    return rows


def non_lateral_false_trigger_report() -> list[dict[str, Any]]:
    rows = []
    holdout_dir = PROJECT_ROOT / "posture_dataset_v2" / "external_holdout" / "holdout_batch_02"
    for path in sorted(holdout_dir.glob("*.csv")):
        if "manifest" in path.name:
            continue
        data = load_csv_playback(path)
        session_v22 = CsvRecognitionSession(data, Recognizer(model_version="v2_2_candidate"))
        session_v24 = CsvRecognitionSession(data, Recognizer(model_version="v2_4_2_candidate"))
        session_v22.process_all()
        session_v24.process_all()
        triggers = [record for record in session_v24.predictions if record.lateral_subclassifier_triggered]
        rows.append({
            "filename": path.name,
            "frame_count": data.frame_count,
            "lateral_trigger_count": len(triggers),
            "lateral_trigger_rate": round(len(triggers) / max(data.frame_count, 1), 6),
            "v2_2_main_posture": session_v22.summary().get("main_posture"),
            "v2_4_2_main_posture": session_v24.summary().get("main_posture"),
            "final_label_changed": session_v22.summary().get("main_posture") != session_v24.summary().get("main_posture"),
        })
    return rows


def leanback_regression_report() -> list[dict[str, Any]]:
    rows = []
    h3_dir = PROJECT_ROOT / "posture_dataset_v2" / "external_holdout" / "v2_2_h3_raw"
    for path in sorted(h3_dir.glob("H3_*.csv")):
        data = load_csv_playback(path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_4_2_candidate"))
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


def object_empty_unknown_report() -> list[dict[str, Any]]:
    rows = []
    object_dir = PROJECT_ROOT / "recognizer" / "object_data" / "batch1_raw"
    for path in sorted(object_dir.glob("object_*.csv")):
        if "manifest" in path.name:
            continue
        data = load_csv_playback(path)
        session = CsvRecognitionSession(data, Recognizer(model_version="v2_4_2_candidate"))
        session.process_all()
        posture_calls = sum(1 for record in session.predictions if record.occupancy_state != "HUMAN" and record.posture)
        lateral_triggers = sum(1 for record in session.predictions if record.lateral_subclassifier_triggered)
        rows.append({"filename": path.name, "frame_count": data.frame_count, "posture_call_count": posture_calls, "lateral_trigger_count": lateral_triggers, "main_status": session.summary().get("main_posture") or "no_posture"})
    api = Recognizer(model_version="v2_4_2_candidate")
    for name, frame in [("__empty_frame__", np.zeros((16, 16), dtype=float)), ("__unknown_low_load__", np.ones((16, 16), dtype=float) * 0.2)]:
        result = api.predict(frame)
        rows.append({"filename": name, "frame_count": 1, "posture_call_count": int(result.get("posture") is not None), "lateral_trigger_count": int(bool(result.get("lateral_subclassifier_triggered"))), "main_status": result.get("occupancy")})
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
        "v2_3_submodel": MODEL_DIR / "lateral_subclassifier_v2_3_candidate.joblib",
        "v2_3_prototype_bank": MODEL_DIR / "lateral_prototype_bank_v2_3_candidate.json",
        "v2_3_runtime_config": MODEL_DIR / "lateral_subclassifier_v2_3_candidate.runtime_config.json",
        "v2_3_bundle": MODEL_DIR / "v2_3_candidate.model_bundle.json",
        "v2_3_1_submodel": MODEL_DIR / "lateral_subclassifier_v2_3_1_candidate.joblib",
        "v2_3_1_prototype_bank": MODEL_DIR / "lateral_prototype_bank_v2_3_1_candidate.json",
        "v2_3_1_runtime_config": MODEL_DIR / "lateral_subclassifier_v2_3_1_candidate.runtime_config.json",
        "v2_3_1_bundle": MODEL_DIR / "v2_3_1_candidate.model_bundle.json",
        "v2_4_submodel": MODEL_DIR / "lateral_merged_subclassifier_v2_4_candidate.joblib",
        "v2_4_metadata": MODEL_DIR / "lateral_merged_subclassifier_v2_4_candidate.metadata.json",
        "v2_4_prototype_bank": MODEL_DIR / "lateral_merged_prototype_bank_v2_4_candidate.json",
        "v2_4_runtime_config": MODEL_DIR / "lateral_merged_subclassifier_v2_4_candidate.runtime_config.json",
        "v2_4_bundle": MODEL_DIR / "v2_4_candidate.model_bundle.json",
        "v2_4_1_submodel": MODEL_DIR / "lateral_merged_subclassifier_v2_4_1_candidate.joblib",
        "v2_4_1_metadata": MODEL_DIR / "lateral_merged_subclassifier_v2_4_1_candidate.metadata.json",
        "v2_4_1_prototype_bank": MODEL_DIR / "lateral_merged_prototype_bank_v2_4_1_candidate.json",
        "v2_4_1_runtime_config": MODEL_DIR / "lateral_merged_subclassifier_v2_4_1_candidate.runtime_config.json",
        "v2_4_1_bundle": MODEL_DIR / "v2_4_1_candidate.model_bundle.json",
        "v2_4_2_submodel": SUBMODEL_PATH,
        "v2_4_2_metadata": SUBMODEL_METADATA_PATH,
        "v2_4_2_prototype_bank": SUBMODEL_PROTOTYPE_PATH,
        "v2_4_2_runtime_config": SUBMODEL_RUNTIME_CONFIG_PATH,
        "v2_4_2_bundle": MODEL_BUNDLE_PATH,
    }
    payload = {key: {"path": str(path), "sha256": sha256_file(path), "exists": path.exists()} for key, path in paths.items()}
    payload["default_model_version"] = default_model_version()
    return payload


def old_hashes(payload: dict[str, Any]) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"default_model", "default_model_version"}:
            kept[key] = value
        elif key.startswith("v2_1") or key.startswith("v2_2") or key.startswith("v2_3"):
            kept[key] = value
        elif key.startswith("v2_4") and not key.startswith("v2_4_2"):
            kept[key] = value
    return kept


def artifact_manifest(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {"model_version": "v2_4_2_candidate", "production_status": "candidate_only", "default_model_unchanged": before.get("default_model") == after.get("default_model"), "default_model_version": default_model_version(), "old_artifacts_unchanged": old_hashes(before) == old_hashes(after), "before": before, "after": after, "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit()}


def generate_plots(samples: list[LateralMergedSample], rows: list[dict[str, Any]], model: LateralMergedFineModelV242) -> None:
    _bar_plot(PLOT_DIR / "lofo_results.png", [row["filename"].replace(".csv", "") for row in rows], [1.0 if row["correct_accept"] else 0.0 for row in rows], "V2.4.2 LOFO correct_accept")
    x, y, _, _ = build_training_matrix(samples)
    _scatter_plot(PLOT_DIR / "boundary_distribution.png", x[:, LATERAL_FEATURE_NAMES.index("left_right_balance")], x[:, LATERAL_FEATURE_NAMES.index("front_back_balance")], list(y), "LR balance vs FB balance")
    distances, labels = [], []
    for sample in samples:
        for features in sample.features:
            pred = model.predict_from_features(features)
            distances.append(float(pred["lateral_prototype_distance"]))
            labels.append(sample.merged_label)
    _scatter_plot(PLOT_DIR / "prototype_distance_distribution.png", np.arange(len(distances), dtype=float), np.asarray(distances, dtype=float), labels, "Prototype distances")
    _confusion_plot(PLOT_DIR / "confusion_matrix.png", rows)
    _bar_plot(PLOT_DIR / "subtype_coverage.png", [s.filename.replace(".csv", "") for s in samples], [1.0 if s.merged_label == SIDE_SITTING_OR_LEANING_LABEL else 0.5 for s in samples], "Subtype coverage: side merged vs diagonal")


def _label_color(label: str) -> tuple[int, int, int]:
    return {SIDE_SITTING_OR_LEANING_LABEL: (47, 128, 237), DIAGONAL_SITTING_LABEL: (242, 153, 74), LATERAL_UNCERTAIN_LABEL: (155, 89, 182)}.get(label, (120, 120, 120))


def _bar_plot(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 1100, 360
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), title, fill=(20, 20, 20))
    margin_left, chart_y, chart_w, chart_h = 50, 42, width - 80, height - 120
    draw.rectangle((margin_left, chart_y, margin_left + chart_w, chart_y + chart_h), outline=(210, 210, 210))
    bar_w = max(8, chart_w // max(len(values), 1) - 6)
    for i, value in enumerate(values):
        x0 = margin_left + i * (chart_w / max(len(values), 1)) + 3
        bar_h = int(chart_h * max(0.0, min(1.0, value)))
        y0 = chart_y + chart_h - bar_h
        draw.rectangle((int(x0), y0, int(x0 + bar_w), chart_y + chart_h), fill=(47, 128, 237) if value >= 0.99 else (242, 153, 74))
        draw.text((int(x0), chart_y + chart_h + 8), labels[i][:14], fill=(60, 60, 60))
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
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _confusion_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    truth_labels = [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL]
    pred_labels = [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL, LATERAL_UNCERTAIN_LABEL]
    matrix = {truth: Counter() for truth in truth_labels}
    for row in rows:
        matrix[row["merged_label"]][row["final_lateral_label"]] += 1
    cell = 110
    img = Image.new("RGB", (560, 360), "white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), "V2.4.2 LOFO confusion matrix", fill=(20, 20, 20))
    ox, oy = 130, 72
    for c, pred in enumerate(pred_labels):
        draw.text((ox + c * cell, oy - 24), pred[:6], fill=(40, 40, 40))
    for r, truth in enumerate(truth_labels):
        draw.text((16, oy + r * cell + 36), truth, fill=(40, 40, 40))
        for c, pred in enumerate(pred_labels):
            val = matrix[truth].get(pred, 0)
            intensity = 245 - min(180, val * 50)
            draw.rectangle((ox + c * cell, oy + r * cell, ox + (c + 1) * cell - 4, oy + (r + 1) * cell - 4), fill=(intensity, intensity, 255), outline=(210, 210, 210))
            draw.text((ox + c * cell + 48, oy + r * cell + 42), str(val), fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_report(samples: list[LateralMergedSample], method: str, comparison: list[dict[str, Any]], lofo: list[dict[str, Any]], subtype: list[dict[str, Any]], diagonal: list[dict[str, Any]], gate_runtime: list[dict[str, Any]], non_lateral: list[dict[str, Any]], leanback: list[dict[str, Any]], object_rows: list[dict[str, Any]], artifacts: dict[str, Any]) -> None:
    lofo_summary = summarize_method(method, lofo)
    subtype_summary = summarize_method("subtype_generalization", subtype)
    diagonal_summary = summarize_method("diagonal_leave_one_file_out", diagonal)
    non_lateral_final_changes = sum(1 for row in non_lateral if bool(row.get("final_label_changed")))
    h3_lateral = sum(int(row["lateral_trigger_count"]) for row in leanback)
    object_triggers = sum(int(row["lateral_trigger_count"]) for row in object_rows)
    object_posture_calls = sum(int(row.get("posture_call_count", 0)) for row in object_rows)
    lines = [
        "# V2.4.2 Lateral Merged Classifier Candidate Report",
        "",
        "V2.4.2 将 `标准侧坐` 与 `侧身倚靠坐` 合并为正式输出 `侧向坐姿`，`斜跨坐` 保持独立，`侧向姿势` 仅作为Boundary安全回退。",
        "",
        f"- Parent model: `v2_2_candidate`",
        f"- Label taxonomy: `{LABEL_TAXONOMY_VERSION}`",
        f"- Selected method: `{method}`",
        f"- LOFO correct_accept: {lofo_summary['correct_accept_count']}/{lofo_summary['file_count']}",
        f"- LOFO correct_fallback: {lofo_summary['correct_fallback_count']}/{lofo_summary['file_count']}",
        f"- LOFO wrong_accept: {lofo_summary['wrong_accept_count']}/{lofo_summary['file_count']}",
        f"- Subtype generalization correct_accept: {subtype_summary['correct_accept_count']}/{subtype_summary['file_count']}, wrong_accept: {subtype_summary['wrong_accept_count']}",
        f"- Diagonal leave-one-file-out correct_accept: {diagonal_summary['correct_accept_count']}/{diagonal_summary['file_count']}, wrong_accept: {diagonal_summary['wrong_accept_count']}",
        f"- Non-lateral final label changes: {non_lateral_final_changes}",
        f"- H3 leanback lateral trigger count: {h3_lateral}",
        f"- Object/EMPTY/UNKNOWN lateral trigger count: {object_triggers}",
        f"- Object posture calls: {object_posture_calls}",
        f"- Default model remains: `{default_model_version()}`",
        f"- Old artifacts unchanged: {artifacts['old_artifacts_unchanged']}",
        "",
        "## Candidate Comparison",
        "",
        "| method | correct_accept | fallback | wrong_accept | safe_resolution | mean boundary |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(f"| {row['method']} | {row['correct_accept_count']}/{row['file_count']} | {row['correct_fallback_count']} | {row['wrong_accept_count']} | {row['safe_resolution_rate']} | {row['mean_boundary_ratio']} |")
    lines.extend([
        "",
        "## Label Mapping",
        "",
        "- `标准侧坐` → `侧向坐姿`",
        "- `侧身倚靠坐` → `侧向坐姿`",
        "- `斜跨坐` → `斜跨坐`",
        "- Boundary fallback → `侧向姿势`",
        "",
        "## Saved Candidate Artifacts",
        "",
        f"- `{SUBMODEL_PATH}`",
        f"- `{SUBMODEL_METADATA_PATH}`",
        f"- `{SUBMODEL_PROTOTYPE_PATH}`",
        f"- `{SUBMODEL_RUNTIME_CONFIG_PATH}`",
        f"- `{MODEL_BUNDLE_PATH}`",
        "",
        "## Required New Holdout",
        "",
        "候选冻结后再录 6 份全新 CSV：正式标签 `侧向坐姿` 4份（原标准侧坐动作2份、原侧身倚靠动作2份）和 `斜跨坐` 2份。该批不得用于训练、Prototype、门控或阈值调整。",
    ])
    (REPORT_DIR / "v2_4_2_candidate_report.md").write_text("\n".join(lines), encoding="utf-8")


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
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
