#!/usr/bin/env python3
"""Export stateful realtime Recognizer configuration for C++ runtime parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognizer.recognizer_api import Recognizer


SCHEMA_VERSION = "runtime_config_export_v1"
DEFAULT_MODEL_VERSION = "v2_4_3_candidate"


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists() or not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _public_payload_keys() -> list[str]:
    return [
        "occupancy",
        "occupancy_confidence",
        "seat_state",
        "posture",
        "posture_confidence",
        "second_label",
        "margin",
        "is_boundary",
        "raw_label",
        "raw_confidence",
        "boundary_reason",
        "prototype_diagnosis",
        "reason",
        "occupancy_features",
        "parent_posture_label",
        "fine_posture_label",
        "final_display_label",
        "subclassifier_triggered",
        "subclassifier_gate_reason",
        "fine_confidence",
        "fine_margin",
        "fine_boundary",
        "fine_boundary_reasons",
        "fine_prototype_label",
        "fine_prototype_distance",
        "fallback_used",
        "parent_model_version",
        "submodel_version",
        "lateral_subclassifier_triggered",
        "lateral_gate_reason",
        "lateral_posture_label",
        "lateral_confidence",
        "lateral_margin",
        "lateral_boundary",
        "lateral_boundary_reasons",
        "lateral_prototype_label",
        "lateral_prototype_distance",
        "lateral_fallback_used",
        "lateral_submodel_version",
        "lateral_second_label",
        "lateral_second_distance",
        "lateral_prototype_margin",
        "lateral_out_of_distribution",
        "lateral_temporal_state",
        "lateral_stable_label",
        "lateral_fallback_requested",
        "lateral_merged_label",
        "lateral_prototype_subtype",
        "lateral_second_subtype",
        "parent_raw_lateral_label",
        "label_taxonomy_version",
        "final_priority_branch",
        "selected_branch",
        "override_reason",
        "fallback_reason",
        "lateral_gate_candidate",
        "lateral_distance_z",
        "lateral_classifier_label",
        "lateral_prototype_source",
        "lateral_second_prototype_source",
        "lateral_normalization_applied",
        "lateral_normalization_reason",
        "lateral_normalization_confidence",
        "lateral_physical_evidence_passed",
        "lateral_physical_evidence_reasons",
        "selected_final_branch",
        "final_override_reason",
        "lateral_gate_strong_evidence",
        "lateral_gate_soft_warnings",
        "lateral_gate_hard_reject_reasons",
        "front_back_support_warning",
        "front_back_support_hard_reject",
        "parent_prototype_agreement",
        "lateral_physical_evidence_score",
        "lateral_gate_decision",
        "lateral_gate_decision_reason",
        "cross_leg_lateral_competition_active",
        "cross_leg_lateral_competition_reason",
        "cross_leg_support_score",
        "lateral_support_score",
        "lateral_vs_cross_leg_margin",
        "conditional_gate_override",
        "conditional_gate_override_reason",
        "final_selected_branch",
    ]


def _lateral_stage_config(stage: object) -> dict[str, object]:
    return {
        "class": stage.__class__.__module__ + "." + stage.__class__.__name__,
        "model_version": getattr(stage, "model_version", None),
        "parent_model_version": getattr(stage, "parent_model_version", None),
        "lateral_submodel_version": getattr(stage, "lateral_submodel_version", None),
        "lateral_hold_frames": getattr(stage, "lateral_hold_frames", None),
        "candidate_lateral_confirm_frames": 3,
        "initial_last_lateral_result": None,
        "initial_missed_lateral_frames": getattr(stage, "_missed_lateral_frames", 0),
        "initial_candidate_lateral_frames": getattr(stage, "_candidate_lateral_frames", 0),
    }


def export_runtime_config(
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    output_path: str | Path | None = None,
    fps: float = 20.0,
    window_seconds: float = 1.5,
    settle_seconds: float = 1.0,
) -> dict[str, object]:
    recognizer = Recognizer(
        model_version=model_version,
        fps=fps,
        window_seconds=window_seconds,
        settle_seconds=settle_seconds,
    )
    analyzer = recognizer._analyzer
    occupancy = analyzer.occupancy_detector
    seat = analyzer.seat_detector
    smoother = analyzer.smoother
    lateral_stage = recognizer._posture_recognizer

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": recognizer.model_version,
        "fps": recognizer.fps,
        "window_seconds": recognizer.window_seconds,
        "settle_seconds": recognizer.settle_seconds,
        "artifact_identity": recognizer.artifact_identity(),
        "seat_analyzer": {
            "class": analyzer.__class__.__module__ + "." + analyzer.__class__.__name__,
            "window_frames": analyzer.window_frames,
            "recent_frames_initial_size": len(analyzer._recent_frames),
            "reset_resets_posture_recognizer": True,
            "reset_preserves_occupancy_detector": True,
        },
        "occupancy_detector": {
            "class": occupancy.__class__.__module__ + "." + occupancy.__class__.__name__,
            "fps": occupancy.fps,
            "empty_total_threshold": occupancy.empty_total_threshold,
            "occupied_total_threshold": occupancy.occupied_total_threshold,
            "human_total_threshold": occupancy.human_total_threshold,
            "detectable_value_threshold": occupancy.detectable_value_threshold,
            "detectable_total_threshold": occupancy.detectable_total_threshold,
            "detectable_points_min": occupancy.detectable_points_min,
            "active_value_threshold": occupancy.active_value_threshold,
            "human_active_area_min": occupancy.human_active_area_min,
            "object_active_area_max": occupancy.object_active_area_max,
            "history_size": occupancy.history.maxlen,
            "use_baseline": occupancy.baseline is not None,
            "baseline_max_frames": None if occupancy.baseline is None else occupancy.baseline.max_frames,
            "baseline_empty_total_threshold": None if occupancy.baseline is None else occupancy.baseline.empty_total_threshold,
        },
        "seat_detector": {
            "class": seat.__class__.__module__ + "." + seat.__class__.__name__,
            "fps": seat.fps,
            "empty_threshold": seat.empty_threshold,
            "occupied_threshold": seat.occupied_threshold,
            "settle_frames": seat.settle_frames,
            "history_size": seat.history_size,
            "total_cv_threshold": seat.total_cv_threshold,
            "cop_motion_threshold": seat.cop_motion_threshold,
            "initial_phase": seat._phase.value,
            "initial_occupied_frames": seat._occupied_frames,
            "initial_stable_frames": seat._stable_frames,
            "initial_empty_frames": seat._empty_frames,
        },
        "smoother": {
            "class": smoother.__class__.__module__ + "." + smoother.__class__.__name__,
            "vote_window": smoother.vote_window,
            "switch_confirmations": smoother.switch_confirmations,
            "min_confidence": smoother.min_confidence,
            "min_margin": smoother.min_margin,
            "uncertain_label": "边界/不确定",
            "initial_current_label": smoother._current_label,
            "initial_pending_label": smoother._pending_label,
            "initial_pending_count": smoother._pending_count,
        },
        "lateral_stage": _lateral_stage_config(lateral_stage),
        "state_rules": {
            "time_dependency": "frame_count_only",
            "classification_input": "mean_of_recent_window_frames",
            "classification_executed_when": "occupancy == HUMAN and seat stable and recent_frame_count == window_frames",
            "non_human_action": "SeatAnalyzer.reset(); occupancy_detector state/history preserved",
            "public_payload_non_human_action": "human-only fields are set to null",
        },
        "public_payload_keys": _public_payload_keys(),
    }
    payload["runtime_config_sha256"] = _sha256_file(recognizer.runtime_config_path)

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return _json_safe(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--window-seconds", type=float, default=1.5)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = export_runtime_config(
        model_version=args.model_version,
        output_path=args.output,
        fps=args.fps,
        window_seconds=args.window_seconds,
        settle_seconds=args.settle_seconds,
    )
    print(f"Exported runtime config for {payload['model_version']} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
