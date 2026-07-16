from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data_loader import DEFAULT_FORMAL_DATASET, default_training_files, read_sensor_csv
from .feature_extractor import summarize_frame
from .occupancy_detector import OccupancyDetector, connected_components
from .rf_recognizer import load_hybrid_recognizer
from .seat_analyzer import SeatAnalyzer


OBJECT_DATA_DIR = Path("recognizer/object_data/batch1_raw")
DEFAULT_OUTPUT_DIR = Path("recognizer/outputs/object_occupancy_batch1")


@dataclass(frozen=True)
class CountingRecognizer:
    recognizer: object
    calls: int = 0

    def predict_posture(self, window: np.ndarray) -> dict[str, object]:
        object.__setattr__(self, "calls", self.calls + 1)
        return self.recognizer.predict_posture(window)


def evaluate_batch(
    object_dir: Path = OBJECT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_path: Path = Path("recognizer/models/rf_posture_v1.joblib"),
    prototype_bank_path: Path = Path("recognizer/models/prototype_bank_v1.json"),
    human_limit: int = 18,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    object_files = sorted(path for path in object_dir.glob("object_*.csv") if path.name != "object_batch1_manifest.csv")
    if not object_files:
        raise FileNotFoundError(f"No object CSV files found in {object_dir}")

    recognizer = load_hybrid_recognizer(model_path, prototype_bank_path)
    quality_rows = [quality_row(path) for path in object_files]
    baseline_rows = [replay_row(path, recognizer) for path in object_files]
    human_rows = [replay_row(path, recognizer, expected_kind="human") for path in _human_regression_files(human_limit)]

    quality_csv = output_dir / "object_batch1_quality.csv"
    replay_csv = output_dir / "object_batch1_replay_current_rules.csv"
    human_csv = output_dir / "human_regression_current_rules.csv"
    summary_json = output_dir / "object_batch1_summary_current_rules.json"

    _write_csv(quality_csv, quality_rows)
    _write_csv(replay_csv, baseline_rows)
    _write_csv(human_csv, human_rows)
    summary = {
        "object_files": len(object_files),
        "object_posture_calls": sum(int(row["posture_model_calls"]) for row in baseline_rows),
        "object_files_with_posture_calls": [row["file"] for row in baseline_rows if int(row["posture_model_calls"]) > 0],
        "object_files_any_human": [row["file"] for row in baseline_rows if int(row["HUMAN_frames"]) > 0],
        "object_files_all_blocked": [
            row["file"] for row in baseline_rows if int(row["posture_model_calls"]) == 0 and int(row["POSTURE_frames"]) == 0
        ],
        "human_files": len(human_rows),
        "human_files_with_posture": [row["file"] for row in human_rows if int(row["POSTURE_frames"]) > 0],
        "human_files_with_object_frames": [row["file"] for row in human_rows if int(row["OBJECT_frames"]) > 0],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "quality_csv": quality_csv,
        "object_replay_csv": replay_csv,
        "human_regression_csv": human_csv,
        "summary_json": summary_json,
    }


def quality_row(path: Path, fps: float = 20.0) -> dict[str, object]:
    _, frames = read_sensor_csv(path)
    totals = frames.sum(axis=(1, 2))
    p95 = float(np.percentile(totals, 95)) if len(totals) else 0.0
    occupied_threshold = max(250.0, p95 * 0.20)
    empty_threshold = 80.0
    occupied_indices = np.flatnonzero(totals >= occupied_threshold)
    if len(occupied_indices):
        start = int(occupied_indices[0])
        end = int(occupied_indices[-1])
        trim = max(1, int(round(0.5 * fps)))
        stable_start = min(start + trim, end)
        stable_end = max(end - trim + 1, stable_start + 1)
        stable = frames[stable_start:stable_end]
    else:
        start = end = stable_start = stable_end = 0
        stable = frames[:0]
    first_empty_s = _continuous_empty_seconds(totals, empty_threshold, from_start=True, fps=fps)
    last_empty_s = _continuous_empty_seconds(totals, empty_threshold, from_start=False, fps=fps)
    stable_totals = stable.sum(axis=(1, 2)) if len(stable) else np.asarray([])
    mean_frame = stable.mean(axis=0) if len(stable) else np.zeros((16, 16))
    active_mask = mean_frame > 15.0
    connected, max_region = connected_components(active_mask)
    summary = summarize_frame(mean_frame)
    cop = np.asarray([(summarize_frame(frame).cop_x, summarize_frame(frame).cop_y) for frame in stable], dtype=float) if len(stable) else np.zeros((0, 2))
    total_drift = _relative_drift(stable_totals)
    return {
        "file": path.name,
        "frame_count": len(frames),
        "front_empty_s": round(first_empty_s, 3),
        "back_empty_s": round(last_empty_s, 3),
        "placement_detected": int(len(occupied_indices) > 0 and start > 0),
        "removal_detected": int(len(occupied_indices) > 0 and end < len(frames) - 1),
        "stable_object_s": round(len(stable) / fps, 3),
        "stable_total_mean": round(float(stable_totals.mean()) if len(stable_totals) else 0.0, 3),
        "stable_total_drift_pct": round(total_drift, 4),
        "active_points": int(active_mask.sum()),
        "active_area": round(float(active_mask.mean()), 4),
        "connected_regions": connected,
        "max_connected_region": max_region,
        "concentration": round(float((mean_frame / max(float(mean_frame.sum()), 1.0)).max()), 4),
        "cop_x": round(summary.cop_x, 3),
        "cop_y": round(summary.cop_y, 3),
        "cop_std": round(float(np.sqrt(cop[:, 0].var() + cop[:, 1].var())) if len(cop) else 0.0, 4),
        "stable_shape": "stable" if total_drift <= 0.35 else "possible_movement_or_drift",
        "quality_ok": int(first_empty_s >= 1.0 and last_empty_s >= 1.0 and len(stable) / fps >= 6.0),
    }


def replay_row(path: Path, recognizer: object, expected_kind: str = "object", fps: float = 20.0) -> dict[str, object]:
    _, frames = read_sensor_csv(path)
    counter = CountingRecognizer(recognizer)
    analyzer = SeatAnalyzer(recognizer=counter, fps=fps, window_seconds=1.5, settle_seconds=1.0)
    occupancy_counts: Counter[str] = Counter()
    seat_counts: Counter[str] = Counter()
    posture_counts: Counter[str] = Counter()
    boundary_count = 0
    for frame in frames:
        result = analyzer.update(frame)
        occupancy_counts[str(result["occupancy_state"])] += 1
        seat_state = "POSTURE" if result.get("posture") else str(result["seat_state"])
        seat_counts[seat_state] += 1
        if result.get("posture"):
            posture_counts[str(result["posture"])] += 1
        boundary_count += int(bool(result.get("is_boundary")))
    return {
        "file": path.name,
        "expected_kind": expected_kind,
        "frame_count": len(frames),
        "EMPTY_frames": occupancy_counts["EMPTY"],
        "LOAD_BELOW_THRESHOLD_frames": occupancy_counts["LOAD_BELOW_THRESHOLD"],
        "OBJECT_frames": occupancy_counts["OBJECT"],
        "HUMAN_frames": occupancy_counts["HUMAN"],
        "UNKNOWN_frames": occupancy_counts["UNKNOWN"],
        "HUMAN_STABILIZING_frames": seat_counts["HUMAN_STABILIZING"],
        "POSTURE_frames": seat_counts["POSTURE"],
        "posture_model_calls": counter.calls,
        "posture_labels": ";".join(f"{label}:{count}" for label, count in posture_counts.most_common()),
        "boundary_frames": boundary_count,
        "blocked_from_posture": int(counter.calls == 0),
    }


def _human_regression_files(limit: int) -> list[Path]:
    preferred = [
        DEFAULT_FORMAL_DATASET / "duanzhengzuozi1-1.csv",
        DEFAULT_FORMAL_DATASET / "duanzhengzuozi1-2.csv",
        DEFAULT_FORMAL_DATASET / "biaozhunkaobeizuo4-1.csv",
        DEFAULT_FORMAL_DATASET / "bantangkaobeizuo5-1.csv",
        DEFAULT_FORMAL_DATASET / "pantuizuo12-1.csv",
        DEFAULT_FORMAL_DATASET / "quansuozuo14-1.csv",
        DEFAULT_FORMAL_DATASET / "biaozhuncezuo15-1.csv",
        DEFAULT_FORMAL_DATASET / "xiekuazuo16-1.csv",
        DEFAULT_FORMAL_DATASET / "ceshenyikaozuo17-1.csv",
    ]
    files = [path for path in preferred if path.exists()]
    for path in default_training_files(include_stage8_10_11=True):
        if path not in files:
            files.append(path)
        if len(files) >= limit:
            break
    return files[:limit]


def _continuous_empty_seconds(totals: np.ndarray, threshold: float, from_start: bool, fps: float) -> float:
    seq = totals if from_start else totals[::-1]
    count = 0
    for value in seq:
        if value <= threshold:
            count += 1
        else:
            break
    return count / fps


def _relative_drift(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(abs(values[-1] - values[0]) / max(float(values.mean()), 1.0))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate object occupancy data against the current occupancy gate.")
    parser.add_argument("--object-dir", default=str(OBJECT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--human-limit", type=int, default=18)
    args = parser.parse_args(argv)
    paths = evaluate_batch(Path(args.object_dir), Path(args.output_dir), human_limit=args.human_limit)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
