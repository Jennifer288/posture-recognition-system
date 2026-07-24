from __future__ import annotations

import csv
import json
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .csv_gui_core import frame_record_from_result, model_export_info
from .frame_orientation import apply_sensor_rotation, orientation_transform_name
from .offline_serial_parser import OfflineSerialParseResult, sha256_file
from .recognizer_api import Recognizer
from .serial_gui_core import ORIENTATION_MODES, apply_orientation


DEFAULT_MODEL_VERSION = "v2_4_3_candidate"
DEFAULT_FPS = 20.0
INITIAL_EMPTY_SECONDS = 3.0
MIN_EMPTY_SECONDS = 1.0
EMPTY_TOTAL_PRESSURE_MAX = 80.0
EMPTY_ACTIVE_POINTS_MAX = 6
MIN_VALID_FRAMES = 20
MIN_STABLE_POSTURE_SECONDS = 2.0
HIGH_BOUNDARY_RATE = 0.40
LOW_MEAN_CONFIDENCE = 0.55
MIXED_POSTURE_SHARE_MAX = 0.60


@dataclass(frozen=True)
class CalibrationInfo:
    calibration_status: str
    calibration_frame_count: int
    calibration_duration_s: float
    calibration_warning: str | None = None


@dataclass(frozen=True)
class OfflineFramePrediction:
    frame_index: int
    time_s: float
    occupancy_state: str
    seat_state: str
    posture: str | None
    posture_confidence: float | None
    raw_label: str | None
    raw_confidence: float | None
    second_label: str | None
    margin: float | None
    is_boundary: bool
    boundary_reason: str | None
    prototype_diagnosis: str | None
    parent_posture_label: str | None
    fine_posture_label: str | None
    lateral_posture_label: str | None
    final_display_label: str | None
    selected_branch: str | None
    inference_ms: float
    total_pressure: float
    max_pressure: float
    active_points: int
    display_status: str


@dataclass(frozen=True)
class OfflinePostureSegment:
    segment_index: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    segment_type: str
    posture: str | None
    mean_confidence: float | None
    boundary_frame_count: int
    frame_count: int


@dataclass(frozen=True)
class OfflineAnalysisSummary:
    result_status: str
    overall_posture: str | None
    total_frames: int
    total_duration_s: float
    human_frames: int
    human_duration_s: float
    stabilizing_frames: int
    stable_posture_frames: int
    stable_posture_duration_s: float
    boundary_frames: int
    boundary_rate: float
    dominant_posture: str | None
    dominant_posture_frames: int
    dominant_posture_duration_s: float
    dominant_posture_share: float
    mean_confidence: float | None
    posture_frame_counts: dict[str, int]
    posture_durations: dict[str, float]
    posture_mean_confidences: dict[str, float | None]
    warnings: list[str] = field(default_factory=list)
    manual_label: str | None = None
    label_matches_overall: bool | None = None


@dataclass(frozen=True)
class OfflineAnalysisResult:
    input_path: Path
    input_type: str
    input_sha256: str
    metadata: dict[str, Any]
    model_version: str
    artifact_info: dict[str, Any]
    fps: float
    fps_source: str
    sensor_rotation_degrees: int
    sensor_orientation_transform: str
    orientation: str
    parser_stats: dict[str, Any]
    invalid_text_lines: list[dict[str, Any]]
    calibration_info: CalibrationInfo
    frames: list[np.ndarray]
    frame_predictions: list[OfflineFramePrediction]
    posture_segments: list[OfflinePostureSegment]
    summary: OfflineAnalysisSummary


ProgressCallback = Callable[[int, int], None]


class OfflinePostureAnalyzer:
    def __init__(
        self,
        *,
        model_version: str = DEFAULT_MODEL_VERSION,
        recognizer_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model_version = model_version
        self.recognizer_factory = recognizer_factory
        self.clock = clock

    def analyze(
        self,
        parse_result: OfflineSerialParseResult,
        *,
        orientation: str | None = None,
        fps: float | None = None,
        sensor_rotation_degrees: int | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> OfflineAnalysisResult:
        metadata = dict(parse_result.selection.metadata)
        resolved_orientation, orientation_warnings = resolve_orientation(metadata, fallback_orientation=orientation)
        resolved_sensor_rotation, rotation_warnings = resolve_sensor_rotation(
            metadata,
            override_rotation=sensor_rotation_degrees,
        )
        resolved_fps, fps_source, fps_warnings = resolve_fps(metadata, manual_fps=fps)
        warnings = list(parse_result.stats.parser_warnings) + orientation_warnings + rotation_warnings + fps_warnings
        sensor_aligned_frames = [apply_sensor_rotation(frame, resolved_sensor_rotation) for frame in parse_result.frames]
        oriented_frames = [apply_orientation(frame, resolved_orientation) for frame in sensor_aligned_frames]

        calibration_info, empty_frames = detect_initial_empty_frames(oriented_frames, resolved_fps)
        if calibration_info.calibration_warning:
            warnings.append(calibration_info.calibration_warning)

        recognizer = self._make_recognizer(resolved_fps)
        _call_if_available(recognizer, "reset")
        if calibration_info.calibration_status == "CALIBRATED_FROM_INITIAL_EMPTY":
            try:
                calibrate = getattr(recognizer, "calibrate")
                calibrate(frames=np.asarray(empty_frames, dtype=np.float32))
            except Exception as exc:
                calibration_info = CalibrationInfo(
                    calibration_status="CALIBRATION_FAILED",
                    calibration_frame_count=0,
                    calibration_duration_s=0.0,
                    calibration_warning=f"空载校准失败: {exc}",
                )
                warnings.append(calibration_info.calibration_warning or "空载校准失败")

        frame_predictions: list[OfflineFramePrediction] = []
        total = len(oriented_frames)
        cancelled = False
        for index, frame in enumerate(oriented_frames):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            start = self.clock()
            raw_result = recognizer.predict(frame)
            inference_ms = max(0.0, (self.clock() - start) * 1000.0)
            record = frame_record_from_result(
                frame=frame,
                result=raw_result,
                frame_index=index,
                timestamp=index / max(resolved_fps, 1e-6),
            )
            frame_predictions.append(_offline_prediction_from_record(record, inference_ms=inference_ms))
            if progress_callback is not None:
                progress_callback(index + 1, total)

        segments = merge_posture_segments(frame_predictions, fps=resolved_fps)
        summary = summarize_analysis(
            frame_predictions,
            segments,
            fps=resolved_fps,
            parser_stats=asdict(parse_result.stats),
            warnings=warnings,
            manual_label=_metadata_label(metadata),
            cancelled=cancelled,
        )
        artifact_info = _safe_model_export_info(recognizer)
        return OfflineAnalysisResult(
            input_path=parse_result.selection.input_path,
            input_type=parse_result.selection.input_type,
            input_sha256=sha256_file(parse_result.selection.input_path),
            metadata=metadata,
            model_version=self.model_version,
            artifact_info=artifact_info,
            fps=resolved_fps,
            fps_source=fps_source,
            sensor_rotation_degrees=resolved_sensor_rotation,
            sensor_orientation_transform=orientation_transform_name(resolved_sensor_rotation),
            orientation=resolved_orientation,
            parser_stats=asdict(parse_result.stats),
            invalid_text_lines=[asdict(item) for item in parse_result.invalid_text_lines],
            calibration_info=calibration_info,
            frames=oriented_frames,
            frame_predictions=frame_predictions,
            posture_segments=segments,
            summary=summary,
        )

    def _make_recognizer(self, fps: float) -> Any:
        if self.recognizer_factory is None:
            return Recognizer(model_version=self.model_version, fps=fps)
        for kwargs in (
            {"model_version": self.model_version, "fps": fps},
            {"fps": fps},
            {},
        ):
            try:
                return self.recognizer_factory(**kwargs)
            except TypeError:
                continue
        return self.recognizer_factory()


def resolve_orientation(metadata: dict[str, Any], *, fallback_orientation: str | None = None) -> tuple[str, list[str]]:
    warnings: list[str] = []
    metadata_orientation = metadata.get("orientation")
    if isinstance(metadata_orientation, str) and metadata_orientation in ORIENTATION_MODES:
        return metadata_orientation, warnings
    if fallback_orientation in ORIENTATION_MODES:
        if metadata_orientation is None:
            warnings.append("未找到方向记录，请确认当前方向设置")
        else:
            warnings.append(f"metadata中的方向记录无效: {metadata_orientation}")
        return str(fallback_orientation), warnings
    warnings.append("未找到方向记录，请确认当前方向设置")
    return "原始", warnings


def resolve_sensor_rotation(
    metadata: dict[str, Any],
    *,
    override_rotation: int | str | None = None,
) -> tuple[int, list[str]]:
    warnings: list[str] = []
    if override_rotation is not None:
        rotation = int(override_rotation)
        orientation_transform_name(rotation)
        return rotation, warnings
    metadata_rotation = metadata.get("sensor_rotation_degrees")
    if metadata_rotation is not None:
        try:
            rotation = int(metadata_rotation)
            orientation_transform_name(rotation)
            return rotation, warnings
        except (TypeError, ValueError):
            warnings.append(f"metadata中的传感器安装方向无效: {metadata_rotation}")
            return 0, warnings
    warnings.append("未找到传感器安装方向记录，按0°旧安装方式分析")
    return 0, warnings


def resolve_fps(metadata: dict[str, Any], *, manual_fps: float | None = None) -> tuple[float, str, list[str]]:
    warnings: list[str] = []
    if manual_fps is not None:
        value = float(manual_fps)
        if value <= 0:
            raise ValueError("FPS必须大于0")
        return value, "manual", warnings
    for key in ("fps", "sample_fps", "sampling_fps"):
        value = metadata.get(key)
        if value is not None and float(value) > 0:
            return float(value), "metadata", warnings
    frames = metadata.get("valid_frames_saved")
    duration = metadata.get("duration_s")
    if frames is not None and duration is not None and float(duration) > 0:
        inferred = float(frames) / float(duration)
        if 1.0 <= inferred <= 100.0:
            return inferred, "metadata_inferred", warnings
        warnings.append(f"metadata推算FPS超出合理范围: {inferred:.4g}")
    warnings.append("未找到采样率，按20 FPS分析")
    return DEFAULT_FPS, "default", warnings


def detect_initial_empty_frames(frames: list[np.ndarray], fps: float) -> tuple[CalibrationInfo, list[np.ndarray]]:
    if not frames:
        return CalibrationInfo("CALIBRATION_SKIPPED", 0, 0.0, "没有有效压力帧，跳过空载校准"), []
    check_count = min(len(frames), max(1, int(round(INITIAL_EMPTY_SECONDS * fps))))
    empty_frames: list[np.ndarray] = []
    for frame in frames[:check_count]:
        arr = np.asarray(frame, dtype=np.float32)
        total = float(arr.sum())
        active = int((arr > 15.0).sum())
        if total <= EMPTY_TOTAL_PRESSURE_MAX and active <= EMPTY_ACTIVE_POINTS_MAX:
            empty_frames.append(arr)
        else:
            break
    min_count = max(1, int(round(MIN_EMPTY_SECONDS * fps)))
    if len(empty_frames) >= min_count:
        return (
            CalibrationInfo(
                "CALIBRATED_FROM_INITIAL_EMPTY",
                len(empty_frames),
                round(len(empty_frames) / max(fps, 1e-6), 4),
                None,
            ),
            empty_frames,
        )
    return (
        CalibrationInfo(
            "NO_RELIABLE_EMPTY_BASELINE",
            len(empty_frames),
            round(len(empty_frames) / max(fps, 1e-6), 4),
            "未找到可靠开头空载段，未执行空载校准",
        ),
        [],
    )


def merge_posture_segments(predictions: list[OfflineFramePrediction], *, fps: float) -> list[OfflinePostureSegment]:
    if not predictions:
        return []
    segments: list[OfflinePostureSegment] = []
    start_index = 0
    current_key = _segment_key(predictions[0])
    for index, record in enumerate(predictions[1:], start=1):
        key = _segment_key(record)
        if key != current_key:
            segments.append(_make_segment(len(segments), predictions[start_index:index], fps=fps))
            start_index = index
            current_key = key
    segments.append(_make_segment(len(segments), predictions[start_index:], fps=fps))
    return segments


def summarize_analysis(
    predictions: list[OfflineFramePrediction],
    segments: list[OfflinePostureSegment],
    *,
    fps: float,
    parser_stats: dict[str, Any],
    warnings: list[str] | None = None,
    manual_label: str | None = None,
    cancelled: bool = False,
) -> OfflineAnalysisSummary:
    warning_items = list(warnings or [])
    total_frames = len(predictions)
    total_duration_s = total_frames / max(fps, 1e-6)
    human_frames = sum(1 for item in predictions if item.occupancy_state == "HUMAN")
    stabilizing_frames = sum(1 for item in segments if item.segment_type == "HUMAN_STABILIZING" for _ in range(item.frame_count))
    boundary_frames = sum(1 for item in predictions if _segment_type(item) == "BOUNDARY")
    posture_records = [item for item in predictions if _segment_type(item) == "POSTURE" and item.posture]
    posture_frame_counts = Counter(item.posture for item in posture_records if item.posture)
    posture_confidences: dict[str, list[float]] = defaultdict(list)
    for item in posture_records:
        if item.posture and item.posture_confidence is not None:
            posture_confidences[item.posture].append(float(item.posture_confidence))
    stable_posture_frames = len(posture_records)
    stable_posture_duration_s = stable_posture_frames / max(fps, 1e-6)
    posture_durations = {label: round(count / max(fps, 1e-6), 4) for label, count in posture_frame_counts.items()}
    posture_mean_confidences = {
        label: round(float(np.mean(values)), 4) if values else None
        for label, values in posture_confidences.items()
    }
    dominant_posture = None
    dominant_frames = 0
    if posture_frame_counts:
        dominant_posture, dominant_frames = posture_frame_counts.most_common(1)[0]
    dominant_duration = dominant_frames / max(fps, 1e-6)
    dominant_share = 0.0 if stable_posture_frames == 0 else dominant_frames / stable_posture_frames
    confidences = [float(item.posture_confidence) for item in posture_records if item.posture_confidence is not None]
    mean_confidence = None if not confidences else round(float(np.mean(confidences)), 4)
    boundary_rate = 0.0 if total_frames == 0 else boundary_frames / total_frames
    result_status = _result_status(
        parser_stats=parser_stats,
        total_frames=total_frames,
        human_frames=human_frames,
        stable_posture_duration_s=stable_posture_duration_s,
        boundary_rate=boundary_rate,
        mean_confidence=mean_confidence,
        dominant_share=dominant_share,
        posture_count=len(posture_frame_counts),
        cancelled=cancelled,
    )
    overall = dominant_posture if result_status == "SUCCESS" else None
    label_matches = None
    if manual_label and overall:
        label_matches = manual_label == overall
    return OfflineAnalysisSummary(
        result_status=result_status,
        overall_posture=overall,
        total_frames=total_frames,
        total_duration_s=round(total_duration_s, 4),
        human_frames=human_frames,
        human_duration_s=round(human_frames / max(fps, 1e-6), 4),
        stabilizing_frames=stabilizing_frames,
        stable_posture_frames=stable_posture_frames,
        stable_posture_duration_s=round(stable_posture_duration_s, 4),
        boundary_frames=boundary_frames,
        boundary_rate=round(boundary_rate, 4),
        dominant_posture=dominant_posture,
        dominant_posture_frames=dominant_frames,
        dominant_posture_duration_s=round(dominant_duration, 4),
        dominant_posture_share=round(dominant_share, 4),
        mean_confidence=mean_confidence,
        posture_frame_counts=dict(posture_frame_counts),
        posture_durations=posture_durations,
        posture_mean_confidences=posture_mean_confidences,
        warnings=warning_items,
        manual_label=manual_label,
        label_matches_overall=label_matches,
    )


def posture_statistics(result: OfflineAnalysisResult) -> list[dict[str, Any]]:
    segment_counts = Counter(segment.posture for segment in result.posture_segments if segment.segment_type == "POSTURE" and segment.posture)
    rows: list[dict[str, Any]] = []
    total = max(result.summary.stable_posture_frames, 1)
    for posture, count in sorted(result.summary.posture_frame_counts.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "posture": posture,
                "frame_count": count,
                "duration_s": result.summary.posture_durations.get(posture, 0.0),
                "share": round(count / total, 4),
                "mean_confidence": result.summary.posture_mean_confidences.get(posture),
                "segment_count": segment_counts.get(posture, 0),
            }
        )
    if result.summary.boundary_frames:
        rows.append(
            {
                "posture": "Boundary",
                "frame_count": result.summary.boundary_frames,
                "duration_s": round(result.summary.boundary_frames / max(result.fps, 1e-6), 4),
                "share": None,
                "mean_confidence": None,
                "segment_count": sum(1 for segment in result.posture_segments if segment.segment_type == "BOUNDARY"),
            }
        )
    return rows


def export_offline_analysis(result: OfflineAnalysisResult, output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = _unique_export_dir(root)
    target.mkdir(parents=True, exist_ok=False)
    report_path = target / "offline_serial_analysis_report.json"
    frame_path = target / "offline_frame_predictions.csv"
    segment_path = target / "offline_posture_segments.csv"
    parse_path = target / "offline_packet_parse_report.json"

    report = {
        "schema_version": "offline_serial_analysis_v1",
        "analysis_time": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(result.input_path),
        "input_type": result.input_type,
        "input_sha256": result.input_sha256,
        "metadata": result.metadata,
        "model_version": result.model_version,
        "model_artifact_identity": result.artifact_info,
        "fps": result.fps,
        "fps_source": result.fps_source,
        "sensor_rotation_degrees": result.sensor_rotation_degrees,
        "sensor_orientation_transform": result.sensor_orientation_transform,
        "orientation": result.orientation,
        "parser_stats": result.parser_stats,
        "invalid_text_lines": result.invalid_text_lines,
        "calibration_info": asdict(result.calibration_info),
        "overall_result": asdict(result.summary),
        "posture_durations": result.summary.posture_durations,
        "posture_frame_counts": result.summary.posture_frame_counts,
        "posture_mean_confidences": result.summary.posture_mean_confidences,
        "warnings": result.summary.warnings,
        "export_completed": True,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    parse_path.write_text(
        json.dumps(
            {
                "schema_version": "offline_packet_parse_v1",
                "input_path": str(result.input_path),
                "input_type": result.input_type,
                "parser_stats": result.parser_stats,
                "invalid_text_lines": result.invalid_text_lines,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with frame_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OfflineFramePrediction.__dataclass_fields__))
        writer.writeheader()
        for record in result.frame_predictions:
            writer.writerow(asdict(record))
    with segment_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OfflinePostureSegment.__dataclass_fields__))
        writer.writeheader()
        for segment in result.posture_segments:
            writer.writerow(asdict(segment))
    return {
        "directory": target,
        "report": report_path,
        "frame_predictions": frame_path,
        "segments": segment_path,
        "parse_report": parse_path,
    }


def _offline_prediction_from_record(record: Any, *, inference_ms: float) -> OfflineFramePrediction:
    return OfflineFramePrediction(
        frame_index=record.frame_index,
        time_s=record.timestamp,
        occupancy_state=record.occupancy_state,
        seat_state=record.seat_state,
        posture=record.posture,
        posture_confidence=record.posture_confidence,
        raw_label=record.raw_label,
        raw_confidence=record.raw_confidence,
        second_label=record.second_label,
        margin=record.margin,
        is_boundary=record.is_boundary,
        boundary_reason=record.boundary_reason,
        prototype_diagnosis=record.prototype_diagnosis,
        parent_posture_label=record.parent_posture_label,
        fine_posture_label=record.fine_posture_label,
        lateral_posture_label=record.lateral_posture_label,
        final_display_label=record.final_display_label,
        selected_branch=record.selected_branch or record.selected_final_branch or record.final_selected_branch,
        inference_ms=round(float(inference_ms), 4),
        total_pressure=record.total_pressure,
        max_pressure=record.max_pressure,
        active_points=record.active_points,
        display_status=record.display_status,
    )


def _segment_type(record: OfflineFramePrediction) -> str:
    if record.occupancy_state == "EMPTY" or record.display_status == "EMPTY":
        return "EMPTY"
    if record.occupancy_state == "OBJECT" or record.display_status == "OBJECT":
        return "OBJECT"
    if record.occupancy_state == "UNKNOWN" or record.display_status == "UNKNOWN":
        return "UNKNOWN"
    if record.is_boundary:
        return "BOUNDARY"
    if record.posture:
        return "POSTURE"
    if record.occupancy_state == "HUMAN":
        return "HUMAN_STABILIZING"
    return "UNKNOWN"


def _segment_key(record: OfflineFramePrediction) -> tuple[str, str | None]:
    kind = _segment_type(record)
    posture = record.posture if kind == "POSTURE" else None
    return kind, posture


def _make_segment(segment_index: int, records: list[OfflineFramePrediction], *, fps: float) -> OfflinePostureSegment:
    first = records[0]
    last = records[-1]
    kind = _segment_type(first)
    confidences = [item.posture_confidence for item in records if item.posture_confidence is not None]
    return OfflinePostureSegment(
        segment_index=segment_index,
        start_frame=first.frame_index,
        end_frame=last.frame_index,
        start_time_s=round(first.time_s, 4),
        end_time_s=round((last.frame_index + 1) / max(fps, 1e-6), 4),
        duration_s=round(len(records) / max(fps, 1e-6), 4),
        segment_type=kind,
        posture=first.posture if kind == "POSTURE" else None,
        mean_confidence=None if not confidences else round(float(np.mean(confidences)), 4),
        boundary_frame_count=sum(1 for item in records if item.is_boundary),
        frame_count=len(records),
    )


def _result_status(
    *,
    parser_stats: dict[str, Any],
    total_frames: int,
    human_frames: int,
    stable_posture_duration_s: float,
    boundary_rate: float,
    mean_confidence: float | None,
    dominant_share: float,
    posture_count: int,
    cancelled: bool,
) -> str:
    if cancelled:
        return "ANALYSIS_CANCELLED"
    if parser_stats.get("valid_packets", 0) <= 0:
        return "INVALID_SERIAL_DATA"
    if human_frames <= 0:
        return "NO_HUMAN"
    if total_frames < MIN_VALID_FRAMES or stable_posture_duration_s < MIN_STABLE_POSTURE_SECONDS:
        return "INSUFFICIENT_DATA"
    if boundary_rate > HIGH_BOUNDARY_RATE:
        return "HIGH_BOUNDARY_RATE"
    if mean_confidence is not None and mean_confidence < LOW_MEAN_CONFIDENCE:
        return "LOW_CONFIDENCE"
    if dominant_share < MIXED_POSTURE_SHARE_MAX and posture_count >= 2:
        return "MIXED_POSTURES"
    return "SUCCESS"


def _safe_model_export_info(recognizer: Any) -> dict[str, Any]:
    try:
        return model_export_info(recognizer)
    except Exception:
        return {"model_version": getattr(recognizer, "model_version", DEFAULT_MODEL_VERSION)}


def _call_if_available(obj: Any, method_name: str) -> None:
    method = getattr(obj, method_name, None)
    if callable(method):
        method()


def _metadata_label(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("label")
    return str(value) if value not in (None, "") else None


def _unique_export_dir(root: Path) -> Path:
    base = root / f"offline_serial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = root / f"{base.name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1
