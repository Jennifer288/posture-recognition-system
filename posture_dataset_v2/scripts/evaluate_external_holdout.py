from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognizer.csv_gui_core import CsvRecognitionSession, FramePrediction, load_csv_playback
from recognizer.recognizer_api import Recognizer


V2_ROOT = PROJECT_ROOT / "posture_dataset_v2"
DEFAULT_HOLDOUT_DIR = V2_ROOT / "external_holdout"
DEFAULT_MANIFEST_TEMPLATE = V2_ROOT / "manifests" / "external_holdout_manifest_template.csv"
DEFAULT_REPORT_ROOT = V2_ROOT / "reports" / "external_holdout"
MODEL_VERSIONS = ("v1", "v2_candidate")
BOUNDARY_LABEL = "Boundary/低置信度"
REQUIRED_MANIFEST_FIELDS = [
    "filename",
    "true_label",
    "holdout_batch",
    "collection_time",
    "data_role",
    "natural_posture",
    "included_in_training",
    "included_in_tuning",
    "notes",
]


RecognizerFactory = Callable[[str], object]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Closed-book V1/V2 external holdout evaluator.")
    parser.add_argument("--holdout-dir", required=True, help="Directory containing external holdout CSV files.")
    parser.add_argument(
        "--manifest",
        help="CSV manifest with true labels. Defaults to <holdout-dir>/external_holdout_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for reports. Defaults to posture_dataset_v2/reports/external_holdout/<timestamp>.",
    )
    parser.add_argument("--fallback-fps", type=float, default=20.0)
    parser.add_argument("--write-template", action="store_true", help="Write the blank manifest template and exit.")
    args = parser.parse_args(argv)

    if args.write_template:
        write_manifest_template(DEFAULT_MANIFEST_TEMPLATE)
        print(json.dumps({"manifest_template": str(DEFAULT_MANIFEST_TEMPLATE)}, ensure_ascii=False, indent=2))
        return 0

    holdout_dir = Path(args.holdout_dir)
    manifest_path = Path(args.manifest) if args.manifest else holdout_dir / "external_holdout_manifest.csv"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_REPORT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    paths = evaluate_external_holdout(
        holdout_dir=holdout_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        fallback_fps=args.fallback_fps,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


def evaluate_external_holdout(
    holdout_dir: Path | str,
    manifest_path: Path | str,
    output_dir: Path | str,
    recognizer_factory: RecognizerFactory | None = None,
    fallback_fps: float = 20.0,
) -> dict[str, Path]:
    holdout = Path(holdout_dir)
    manifest = Path(manifest_path)
    output = Path(output_dir)
    recognizer_factory = recognizer_factory or (lambda model_version: Recognizer(model_version=model_version))

    samples = read_holdout_manifest(manifest, holdout)
    validate_directory_csvs_are_manifested(holdout, manifest, samples)
    output.mkdir(parents=True, exist_ok=True)

    file_rows_by_version: dict[str, list[dict[str, Any]]] = {}
    frame_rows_by_version: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    artifact_hashes = model_artifact_hashes()

    for model_version in MODEL_VERSIONS:
        file_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        recognizer = recognizer_factory(model_version)
        for sample in samples:
            playback = load_csv_playback(sample["path"], fallback_fps=fallback_fps)
            reset = getattr(recognizer, "reset", None)
            if callable(reset):
                reset()
            session = CsvRecognitionSession(playback, recognizer)
            records = session.process_all()
            file_row = summarize_file(sample, playback.frame_count, records, session.summary(), model_version, recognizer)
            file_rows.append(file_row)
            frame_rows.extend(frame_prediction_rows(sample, model_version, records))
        file_rows_by_version[model_version] = file_rows
        frame_rows_by_version[model_version] = frame_rows
        summaries[model_version] = model_summary(model_version, file_rows)

    paths = {
        "holdout_v1_file_predictions": output / "holdout_v1_file_predictions.csv",
        "holdout_v2_file_predictions": output / "holdout_v2_file_predictions.csv",
        "holdout_v1_frame_predictions": output / "holdout_v1_frame_predictions.csv",
        "holdout_v2_frame_predictions": output / "holdout_v2_frame_predictions.csv",
        "holdout_model_comparison": output / "holdout_model_comparison.csv",
        "holdout_per_class_recall": output / "holdout_per_class_recall.csv",
        "holdout_confusion_matrices": output / "holdout_confusion_matrices.json",
        "holdout_final_report": output / "holdout_final_report.md",
        "holdout_artifact_hashes": output / "holdout_artifact_hashes.json",
    }
    write_csv(paths["holdout_v1_file_predictions"], file_rows_by_version["v1"])
    write_csv(paths["holdout_v2_file_predictions"], file_rows_by_version["v2_candidate"])
    write_csv(paths["holdout_v1_frame_predictions"], frame_rows_by_version["v1"])
    write_csv(paths["holdout_v2_frame_predictions"], frame_rows_by_version["v2_candidate"])
    comparison_rows = model_comparison_rows(file_rows_by_version["v1"], file_rows_by_version["v2_candidate"])
    write_csv(paths["holdout_model_comparison"], comparison_rows)
    write_csv(paths["holdout_per_class_recall"], per_class_recall_rows(file_rows_by_version))
    paths["holdout_confusion_matrices"].write_text(
        json.dumps(confusion_payload(file_rows_by_version), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["holdout_artifact_hashes"].write_text(json.dumps(artifact_hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["holdout_final_report"].write_text(
        render_report(samples, summaries, comparison_rows, artifact_hashes),
        encoding="utf-8",
    )
    return paths


def write_manifest_template(path: Path | str = DEFAULT_MANIFEST_TEMPLATE) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_MANIFEST_FIELDS)
        writer.writeheader()
    return target


def read_holdout_manifest(path: Path, holdout_dir: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"External holdout manifest not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing required fields: {missing}")
        rows = []
        seen: set[str] = set()
        for row in reader:
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            if filename in seen:
                raise ValueError(f"Duplicate manifest filename: {filename}")
            seen.add(filename)
            if truthy(row.get("included_in_training")) or truthy(row.get("included_in_tuning")):
                raise ValueError(f"Holdout row must not be included in training/tuning: {filename}")
            if str(row.get("data_role", "")).strip() != "external_holdout":
                raise ValueError(f"data_role must be external_holdout for {filename}")
            true_label = str(row.get("true_label", "")).strip()
            if not true_label:
                raise ValueError(f"true_label is required for {filename}")
            csv_path = holdout_dir / filename
            if not csv_path.exists():
                raise FileNotFoundError(f"Manifest CSV is missing from holdout directory: {csv_path}")
            item = dict(row)
            item["filename"] = filename
            item["true_label"] = true_label
            item["path"] = csv_path
            rows.append(item)
    if not rows:
        raise ValueError(f"Manifest has no holdout CSV rows: {path}")
    return rows


def validate_directory_csvs_are_manifested(holdout_dir: Path, manifest_path: Path, samples: Sequence[dict[str, Any]]) -> None:
    if not holdout_dir.exists():
        raise FileNotFoundError(f"Holdout directory not found: {holdout_dir}")
    manifested = {sample["filename"] for sample in samples}
    csvs = {
        path.name
        for path in holdout_dir.glob("*.csv")
        if path.resolve() != manifest_path.resolve()
    }
    extra = sorted(csvs - manifested)
    if extra:
        raise ValueError(f"CSV files are present but missing true labels in manifest: {extra}")


def summarize_file(
    sample: dict[str, Any],
    total_frames: int,
    records: Sequence[FramePrediction],
    summary: dict[str, Any],
    model_version: str,
    recognizer: object,
) -> dict[str, Any]:
    processed = len(records)
    export_complete = processed == total_frames
    valid = export_complete
    posture_labels = [record.posture for record in records if record.posture]
    boundary_frames = [record for record in records if record.display_status == "POSTURE" and record.is_boundary]
    raw_labels = [record.raw_label for record in records if record.raw_label]
    confidences = [record.posture_confidence for record in records if record.posture_confidence is not None]
    raw_confidences = [record.raw_confidence for record in records if record.raw_confidence is not None]
    margins = [record.margin for record in records if record.margin is not None]
    second_labels = [record.second_label for record in records if record.second_label]
    final_label = majority(posture_labels)
    if final_label is None:
        final_label = BOUNDARY_LABEL if boundary_frames else majority([record.display_status for record in records]) or ""
    raw_label = majority(raw_labels) or ""
    true_label = sample["true_label"]
    correct = bool(valid and final_label == true_label)
    raw_correct = bool(valid and raw_label == true_label)
    wrong_accept = bool(valid and final_label not in {"", BOUNDARY_LABEL} and final_label != true_label)
    correct_reject = bool(valid and final_label == BOUNDARY_LABEL and raw_label == true_label)
    return {
        "model_version": model_version,
        "filename": sample["filename"],
        "source_path": str(sample["path"]),
        "true_label": true_label,
        "holdout_batch": sample.get("holdout_batch", ""),
        "collection_time": sample.get("collection_time", ""),
        "natural_posture": sample.get("natural_posture", ""),
        "final_label": final_label,
        "correct": correct,
        "boundary_ratio": round(sum(1 for record in records if record.is_boundary) / max(processed, 1), 6),
        "rf_raw_first": raw_label,
        "raw_correct": raw_correct,
        "first_stable_output_delay_s": summary.get("first_posture_delay_s"),
        "label_switch_count": summary.get("label_switch_count"),
        "confidence_mean": optional_round(np.mean(confidences) if confidences else None),
        "confidence_min": optional_round(np.min(confidences) if confidences else None),
        "raw_confidence_mean": optional_round(np.mean(raw_confidences) if raw_confidences else None),
        "second_label": majority(second_labels) or "",
        "margin_mean": optional_round(np.mean(margins) if margins else None),
        "margin_min": optional_round(np.min(margins) if margins else None),
        "processed_frames": processed,
        "csv_total_frames": total_frames,
        "export_complete": export_complete,
        "valid_evaluation": valid,
        "invalid_reason": "" if valid else f"processed_frames({processed}) != csv_total_frames({total_frames})",
        "wrong_accept": wrong_accept,
        "correct_but_rejected": correct_reject,
        "model_path": str(getattr(recognizer, "model_path", "")),
        "prototype_bank_path": str(getattr(recognizer, "prototype_bank_path", "")),
        "notes": sample.get("notes", ""),
    }


def frame_prediction_rows(sample: dict[str, Any], model_version: str, records: Sequence[FramePrediction]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = asdict(record)
        row.update(
            {
                "model_version": model_version,
                "filename": sample["filename"],
                "true_label": sample["true_label"],
                "frame_final_correct": bool(record.posture == sample["true_label"]) if record.posture else False,
                "frame_raw_correct": bool(record.raw_label == sample["true_label"]) if record.raw_label else False,
            }
        )
        rows.append(row)
    return rows


def model_summary(model_version: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_evaluation"]]
    total = len(valid)
    correct = sum(1 for row in valid if row["correct"])
    raw_correct = sum(1 for row in valid if row["raw_correct"])
    return {
        "model_version": model_version,
        "valid_files": total,
        "invalid_files": len(rows) - total,
        "boundary_aware_file_accuracy": round(correct / total, 6) if total else None,
        "raw_file_accuracy": round(raw_correct / total, 6) if total else None,
        "mean_boundary_ratio": optional_round(np.mean([row["boundary_ratio"] for row in valid]) if valid else None),
        "wrong_accept_count": sum(1 for row in valid if row["wrong_accept"]),
        "correct_but_rejected_count": sum(1 for row in valid if row["correct_but_rejected"]),
    }


def model_comparison_rows(v1_rows: Sequence[dict[str, Any]], v2_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_v2 = {row["filename"]: row for row in v2_rows}
    rows = []
    for v1 in v1_rows:
        v2 = by_v2[v1["filename"]]
        if not v1["valid_evaluation"] or not v2["valid_evaluation"]:
            change = "invalid"
        elif v2["correct"] and not v1["correct"]:
            change = "improved"
        elif v1["correct"] and not v2["correct"]:
            change = "regressed"
        else:
            change = "same"
        rows.append(
            {
                "filename": v1["filename"],
                "true_label": v1["true_label"],
                "v1_final_label": v1["final_label"],
                "v1_correct": v1["correct"],
                "v1_raw_first": v1["rf_raw_first"],
                "v1_raw_correct": v1["raw_correct"],
                "v1_boundary_ratio": v1["boundary_ratio"],
                "v2_final_label": v2["final_label"],
                "v2_correct": v2["correct"],
                "v2_raw_first": v2["rf_raw_first"],
                "v2_raw_correct": v2["raw_correct"],
                "v2_boundary_ratio": v2["boundary_ratio"],
                "change": change,
                "v1_valid": v1["valid_evaluation"],
                "v2_valid": v2["valid_evaluation"],
            }
        )
    return rows


def per_class_recall_rows(rows_by_version: dict[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for model_version, file_rows in rows_by_version.items():
        labels = sorted({row["true_label"] for row in file_rows})
        for label in labels:
            valid = [row for row in file_rows if row["true_label"] == label and row["valid_evaluation"]]
            total = len(valid)
            correct = sum(1 for row in valid if row["correct"])
            raw_correct = sum(1 for row in valid if row["raw_correct"])
            rows.append(
                {
                    "model_version": model_version,
                    "true_label": label,
                    "valid_files": total,
                    "correct_files": correct,
                    "recall": round(correct / total, 6) if total else None,
                    "raw_correct_files": raw_correct,
                    "raw_recall": round(raw_correct / total, 6) if total else None,
                    "boundary_files": sum(1 for row in valid if row["final_label"] == BOUNDARY_LABEL),
                    "mean_boundary_ratio": optional_round(np.mean([row["boundary_ratio"] for row in valid]) if valid else None),
                }
            )
    return rows


def confusion_payload(rows_by_version: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for model_version, rows in rows_by_version.items():
        valid = [row for row in rows if row["valid_evaluation"]]
        payload[model_version] = {
            "boundary_aware": confusion(valid, "final_label"),
            "raw": confusion(valid, "rf_raw_first"),
            "invalid_files": [row["filename"] for row in rows if not row["valid_evaluation"]],
        }
    return payload


def confusion(rows: Sequence[dict[str, Any]], prediction_key: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        matrix[str(row["true_label"])][str(row.get(prediction_key, ""))] += 1
    return {truth: dict(preds) for truth, preds in sorted(matrix.items())}


def render_report(
    samples: Sequence[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    comparison_rows: Sequence[dict[str, Any]],
    artifact_hashes: dict[str, str],
) -> str:
    lines = [
        "# External Holdout Closed-Book Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        f"- Holdout files evaluated: {len(samples)}",
        "- No training, tuning, threshold adjustment, metadata update, or model overwrite is performed by this script.",
        "- Files with incomplete processing are marked invalid and excluded from accuracy.",
        "",
        "## Summary",
        "",
        "| Model | Valid Files | Boundary-Aware Accuracy | RF Raw Accuracy | Boundary Ratio | Wrong Accept | Correct But Rejected |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_version in MODEL_VERSIONS:
        row = summaries[model_version]
        lines.append(
            f"| {model_version} | {row['valid_files']} | {format_metric(row['boundary_aware_file_accuracy'])} | "
            f"{format_metric(row['raw_file_accuracy'])} | {format_metric(row['mean_boundary_ratio'])} | "
            f"{row['wrong_accept_count']} | {row['correct_but_rejected_count']} |"
        )
    improved = [row["filename"] for row in comparison_rows if row["change"] == "improved"]
    regressed = [row["filename"] for row in comparison_rows if row["change"] == "regressed"]
    lines.extend(
        [
            "",
            "## V2 Compared With V1",
            "",
            f"- Improved files: {', '.join(improved) if improved else 'None'}",
            f"- Regressed files: {', '.join(regressed) if regressed else 'None'}",
            "",
            "## Artifact Hashes",
            "",
        ]
    )
    for path, digest in sorted(artifact_hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def model_artifact_hashes() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "recognizer" / "models" / "rf_posture_v1.joblib",
        PROJECT_ROOT / "recognizer" / "models" / "rf_posture_v1.metadata.json",
        PROJECT_ROOT / "recognizer" / "models" / "prototype_bank_v1.json",
        PROJECT_ROOT / "recognizer" / "models" / "rf_posture_v2_candidate.joblib",
        PROJECT_ROOT / "recognizer" / "models" / "rf_posture_v2_candidate.metadata.json",
        PROJECT_ROOT / "recognizer" / "models" / "prototype_bank_v2_candidate.json",
        PROJECT_ROOT / "recognizer" / "models" / "rf_posture_v2_candidate.runtime_config.json",
    ]
    return {str(path): sha256_file(path) for path in paths if path.exists()}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def majority(values: Iterable[str | None]) -> str | None:
    clean = [value for value in values if value]
    if not clean:
        return None
    return Counter(clean).most_common(1)[0][0]


def optional_round(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def format_metric(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
