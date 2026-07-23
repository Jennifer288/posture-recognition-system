#!/usr/bin/env python3
"""Export stateful Recognizer.predict golden records for C++ runtime parity."""

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

from recognizer.data_loader import read_sensor_csv
from recognizer.recognizer_api import Recognizer


SCHEMA_VERSION = "runtime_recognizer_golden_v1"
DEFAULT_MODEL_VERSION = "v2_4_3_candidate"


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _seat_state(analyzer: object) -> dict[str, object]:
    seat = analyzer.seat_detector
    return {
        "phase": seat._phase.value,
        "occupied_frames": seat._occupied_frames,
        "stable_frames": seat._stable_frames,
        "empty_frames": seat._empty_frames,
        "recent_totals": list(seat._recent_totals),
        "recent_cop": [list(item) for item in seat._recent_cop],
    }


def _smoother_state(analyzer: object) -> dict[str, object]:
    smoother = analyzer.smoother
    return {
        "recent": list(smoother._recent),
        "current_label": smoother._current_label,
        "pending_label": smoother._pending_label,
        "pending_count": smoother._pending_count,
    }


def _lateral_state(recognizer: Recognizer) -> dict[str, object]:
    stage = recognizer._posture_recognizer
    return {
        "last_lateral_result_present": getattr(stage, "_last_lateral_result", None) is not None,
        "missed_lateral_frames": getattr(stage, "_missed_lateral_frames", None),
        "candidate_lateral_frames": getattr(stage, "_candidate_lateral_frames", None),
        "lateral_hold_frames": getattr(stage, "lateral_hold_frames", None),
    }


def _analyzer_state(analyzer: object) -> dict[str, object]:
    return {
        "window_frames": analyzer.window_frames,
        "recent_frame_count": len(analyzer._recent_frames),
        "recent_frame_totals": [float(item.sum()) for item in analyzer._recent_frames],
    }


def _classification_executed(payload: dict[str, Any]) -> bool:
    return payload.get("occupancy") == "HUMAN" and payload.get("seat_state") == "HUMAN_RECOGNIZING"


def _classification_skip_reason(payload: dict[str, Any]) -> str | None:
    if _classification_executed(payload):
        return None
    if payload.get("occupancy") != "HUMAN":
        return str(payload.get("reason") or "non_human_occupancy")
    return str(payload.get("reason") or "human_waiting_for_stable_window")


def export_runtime_recognizer_golden(
    pressure_frames_path: str | Path,
    output_path: str | Path,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    fps: float = 20.0,
    window_seconds: float = 1.5,
    settle_seconds: float = 1.0,
) -> int:
    timestamps, frames = read_sensor_csv(pressure_frames_path)
    recognizer = Recognizer(
        model_version=model_version,
        fps=fps,
        window_seconds=window_seconds,
        settle_seconds=settle_seconds,
    )
    recognizer.reset()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for frame_index, frame in enumerate(frames):
            frame_uint8 = _uint8_frame(frame, label=f"frame {frame_index}")
            analyzer = recognizer._analyzer
            seat_before = _seat_state(analyzer)
            smoother_before = _smoother_state(analyzer)
            lateral_before = _lateral_state(recognizer)
            payload = recognizer.predict(frame)
            analyzer_after = recognizer._analyzer
            classification_executed = _classification_executed(payload)
            record = {
                "schema_version": SCHEMA_VERSION,
                "frame_index": frame_index,
                "timestamp": timestamps[frame_index] if frame_index < len(timestamps) else None,
                "model_version": model_version,
                "fps": fps,
                "frame_uint8": frame_uint8.reshape(-1).astype(int).tolist(),
                "frame_uint8_sha256": _sha256_bytes(frame_uint8.tobytes()),
                "public_payload": payload,
                "occupancy_debug": {
                    "state": payload.get("occupancy"),
                    "confidence": payload.get("occupancy_confidence"),
                    "reason": payload.get("reason"),
                    "features": payload.get("occupancy_features"),
                },
                "seat_debug": {
                    "before": seat_before,
                    "after": _seat_state(analyzer_after),
                    "seat_state": payload.get("seat_state"),
                },
                "analyzer_state": _analyzer_state(analyzer_after),
                "stable_gate": {
                    "is_stable": payload.get("seat_state") == "HUMAN_RECOGNIZING",
                    "window_full": len(analyzer_after._recent_frames) == analyzer_after.window_frames,
                    "classification_executed": classification_executed,
                    "skip_reason": _classification_skip_reason(payload),
                },
                "classification_executed": classification_executed,
                "parent_stage": {
                    "label": payload.get("parent_posture_label"),
                    "confidence": payload.get("posture_confidence"),
                    "second_label": payload.get("second_label"),
                    "margin": payload.get("margin"),
                },
                "leanback_stage": {
                    "triggered": payload.get("subclassifier_triggered"),
                    "gate_reason": payload.get("subclassifier_gate_reason"),
                    "fine_label": payload.get("fine_posture_label"),
                    "fine_confidence": payload.get("fine_confidence"),
                    "fallback_used": payload.get("fallback_used"),
                },
                "lateral_stage": {
                    "before": lateral_before,
                    "after": _lateral_state(recognizer),
                    "triggered": payload.get("lateral_subclassifier_triggered"),
                    "gate_reason": payload.get("lateral_gate_reason"),
                    "label": payload.get("lateral_posture_label"),
                    "confidence": payload.get("lateral_confidence"),
                    "temporal_state": payload.get("lateral_temporal_state"),
                    "final_selected_branch": payload.get("final_selected_branch"),
                },
                "smoother_debug": {
                    "before": smoother_before,
                    "after": _smoother_state(analyzer_after),
                    "label": payload.get("posture"),
                    "raw_label": payload.get("raw_label"),
                    "raw_confidence": payload.get("raw_confidence"),
                    "is_boundary": payload.get("is_boundary"),
                    "boundary_reason": payload.get("boundary_reason"),
                },
            }
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-frames", required=True, type=Path)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--window-seconds", type=float, default=1.5)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    count = export_runtime_recognizer_golden(
        args.pressure_frames,
        args.output,
        model_version=args.model_version,
        fps=args.fps,
        window_seconds=args.window_seconds,
        settle_seconds=args.settle_seconds,
    )
    print(f"Exported {count} runtime golden records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
