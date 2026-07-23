#!/usr/bin/env python3
"""Export per-frame Python recognizer golden data for C/C++ parity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognizer.data_loader import read_sensor_csv
from recognizer.feature_extractor import FEATURE_DIM, FEATURE_NAMES, as_frame, extract_features, summarize_frame
from recognizer.occupancy_detector import OccupancyDetector, OccupancyResult
from recognizer.recognizer_api import Recognizer


DEFAULT_MODEL_VERSION = "v2_4_3_candidate"
ALGORITHM_ORIENTATION = "raw_input_no_runtime_transform"
FEATURE_ACTIVE_THRESHOLD = 20.0


def _uint8_frame(frame: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(frame, dtype=float)
    if arr.shape != (16, 16):
        raise ValueError(f"{label} must be shaped (16,16), got {arr.shape}")
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded, atol=0.0, rtol=0.0):
        raise ValueError(f"{label} contains non-integer pressure values")
    if float(rounded.min()) < 0.0 or float(rounded.max()) > 255.0:
        raise ValueError(f"{label} contains values outside uint8 range 0..255")
    return np.ascontiguousarray(rounded.astype(np.uint8))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _float64_sha256(values: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(values, dtype=np.float64).tobytes())


def _occupancy_dict(result: OccupancyResult) -> dict[str, object]:
    return {
        "state": result.state.value,
        "confidence": result.confidence,
        "reason": result.reason,
        "total_pressure": result.total_pressure,
        "detectable_points": result.detectable_points,
        "detectable_area": result.detectable_area,
        "active_area": result.active_area,
        "connected_regions": result.connected_regions,
        "pressure_spread": result.pressure_spread,
        "active_points": result.active_points,
        "max_region_area": result.max_region_area,
        "concentration": result.concentration,
        "left_right_extent": result.left_right_extent,
        "front_back_extent": result.front_back_extent,
        "total_cv": result.total_cv,
        "cop_motion": result.cop_motion,
        "gradual_loading": result.gradual_loading,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_golden_records(
    pressure_frames_path: str | Path,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    fps: float = 20.0,
) -> Iterable[dict[str, object]]:
    """Yield JSON-serializable golden records without changing recognizer behavior."""

    timestamps, frames = read_sensor_csv(pressure_frames_path)
    recognizer = Recognizer(model_version=model_version, fps=fps)
    recognizer.reset()
    realtime_occupancy = OccupancyDetector(fps=fps)
    static_occupancy = OccupancyDetector(fps=fps, use_baseline=False)
    artifact_identity = recognizer.artifact_identity()
    feature_names = list(FEATURE_NAMES)

    for frame_index, frame in enumerate(frames):
        raw_uint8 = _uint8_frame(frame, label=f"frame {frame_index}")
        algorithm_frame = as_frame(frame)
        algorithm_uint8 = _uint8_frame(algorithm_frame, label=f"algorithm frame {frame_index}")
        summary = summarize_frame(algorithm_frame)
        features = np.asarray(extract_features(algorithm_frame), dtype=np.float64)
        normalized = np.ascontiguousarray(features[:256].reshape(16, 16), dtype=np.float64)
        realtime_occ = realtime_occupancy.update(algorithm_frame)
        static_occ = static_occupancy.analyze(algorithm_frame)
        prediction = recognizer.predict(algorithm_frame)

        total = float(algorithm_frame.sum())
        left_pressure = float(algorithm_frame[:, :8].sum())
        right_pressure = float(algorithm_frame[:, 8:].sum())
        front_pressure = float(algorithm_frame[:8, :].sum())
        back_pressure = float(algorithm_frame[8:, :].sum())

        yield {
            "schema_version": "recognizer_golden_v1",
            "frame_index": frame_index,
            "timestamp": timestamps[frame_index] if frame_index < len(timestamps) else None,
            "model_version": model_version,
            "artifact_identity": artifact_identity,
            "input_dtype_after_as_frame": str(algorithm_frame.dtype),
            "input_shape": [16, 16],
            "input_value_range": [int(raw_uint8.min()), int(raw_uint8.max())],
            "input_matrix_direction": ALGORITHM_ORIENTATION,
            "raw_frame": raw_uint8.astype(int).tolist(),
            "raw_frame_flat": raw_uint8.reshape(-1).astype(int).tolist(),
            "raw_frame_uint8_sha256": _sha256_bytes(raw_uint8.tobytes()),
            "algorithm_frame": algorithm_uint8.astype(int).tolist(),
            "algorithm_frame_flat": algorithm_uint8.reshape(-1).astype(int).tolist(),
            "algorithm_frame_uint8_sha256": _sha256_bytes(algorithm_uint8.tobytes()),
            "static_occupancy": _occupancy_dict(static_occ),
            "static_occupancy_state": static_occ.state.value,
            "realtime_occupancy": _occupancy_dict(realtime_occ),
            "realtime_occupancy_state": realtime_occ.state.value,
            "total_pressure": total,
            "max_pressure": float(algorithm_frame.max()),
            "min_pressure": float(algorithm_frame.min()),
            "mean_pressure": float(algorithm_frame.mean()),
            "active_points": int((algorithm_frame > FEATURE_ACTIVE_THRESHOLD).sum()),
            "feature_active_points": int((algorithm_frame > FEATURE_ACTIVE_THRESHOLD).sum()),
            "active_point_threshold": FEATURE_ACTIVE_THRESHOLD,
            "cop_col": summary.cop_x,
            "cop_row": summary.cop_y,
            "left_pressure": left_pressure,
            "right_pressure": right_pressure,
            "front_pressure": front_pressure,
            "back_pressure": back_pressure,
            "left_share": summary.left_share,
            "right_share": summary.right_share,
            "front_share": summary.front_share,
            "back_share": summary.back_share,
            "left_right_balance": (left_pressure - right_pressure) / (total if total > 0.0 else 1.0),
            "front_back_balance": (front_pressure - back_pressure) / (total if total > 0.0 else 1.0),
            "normalized_flat": normalized.reshape(-1).tolist(),
            "normalized_float64_sha256": _float64_sha256(normalized),
            "features": features.tolist(),
            "feature_dim": FEATURE_DIM,
            "feature_names": feature_names,
            "python_final_label": prediction.get("final_display_label") or prediction.get("posture"),
            "python_final_confidence": prediction.get("posture_confidence"),
            "python_boundary": prediction.get("is_boundary"),
            "python_boundary_reason": prediction.get("boundary_reason"),
            "python_selected_branch": prediction.get("selected_branch"),
            "python_final_payload": _json_safe(prediction),
        }


def export_golden(
    pressure_frames_path: str | Path,
    output_path: str | Path,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    fps: float = 20.0,
) -> int:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in build_golden_records(pressure_frames_path, model_version=model_version, fps=fps):
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-frames", required=True, type=Path, help="Input pressure_frames.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL golden file")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION, help="Recognizer model version")
    parser.add_argument("--fps", type=float, default=20.0, help="Recognizer FPS")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    count = export_golden(args.pressure_frames, args.output, model_version=args.model_version, fps=args.fps)
    print(f"Exported {count} golden frame records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
