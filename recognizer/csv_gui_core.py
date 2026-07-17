from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import read_sensor_csv
from .feature_extractor import as_frame
from .recognizer_api import Recognizer


class CsvGuiError(RuntimeError):
    pass


class CsvFormatError(CsvGuiError):
    pass


@dataclass(frozen=True)
class CsvPlaybackData:
    path: Path
    timestamps: list[str]
    frames: np.ndarray
    frame_times_s: list[float]
    fps: float
    time_source: str

    @property
    def frame_count(self) -> int:
        return int(len(self.frames))

    @property
    def duration_s(self) -> float:
        if not self.frame_times_s:
            return 0.0
        if len(self.frame_times_s) == 1:
            return 1.0 / max(self.fps, 1e-6)
        return float(self.frame_times_s[-1] + self.frame_interval_s(len(self.frame_times_s) - 1))

    def frame_interval_s(self, index: int) -> float:
        if len(self.frame_times_s) >= 2 and 0 <= index < len(self.frame_times_s) - 1:
            interval = self.frame_times_s[index + 1] - self.frame_times_s[index]
            if interval > 0:
                return float(interval)
        return 1.0 / max(self.fps, 1e-6)


@dataclass(frozen=True)
class FramePrediction:
    timestamp: float
    frame_index: int
    occupancy_state: str
    occupancy_confidence: float | None
    seat_state: str
    display_status: str
    posture: str | None
    posture_confidence: float | None
    raw_label: str | None
    raw_confidence: float | None
    second_label: str | None
    margin: float | None
    is_boundary: bool
    boundary_reason: str | None
    prototype_diagnosis: str | None
    parent_posture_label: str | None = None
    fine_posture_label: str | None = None
    final_display_label: str | None = None
    subclassifier_triggered: bool | None = None
    subclassifier_gate_reason: str | None = None
    fine_confidence: float | None = None
    fine_margin: float | None = None
    fine_boundary: bool | None = None
    fine_boundary_reasons: str | None = None
    fine_prototype_label: str | None = None
    fine_prototype_distance: float | None = None
    fallback_used: bool | None = None
    parent_model_version: str | None = None
    submodel_version: str | None = None
    lateral_subclassifier_triggered: bool | None = None
    lateral_gate_reason: str | None = None
    lateral_posture_label: str | None = None
    lateral_confidence: float | None = None
    lateral_margin: float | None = None
    lateral_boundary: bool | None = None
    lateral_boundary_reasons: str | None = None
    lateral_prototype_label: str | None = None
    lateral_prototype_distance: float | None = None
    lateral_fallback_used: bool | None = None
    lateral_submodel_version: str | None = None
    lateral_second_label: str | None = None
    lateral_second_distance: float | None = None
    lateral_prototype_margin: float | None = None
    lateral_out_of_distribution: bool | None = None
    lateral_temporal_state: str | None = None
    lateral_stable_label: str | None = None
    lateral_fallback_requested: bool | None = None
    final_priority_branch: str | None = None
    selected_branch: str | None = None
    override_reason: str | None = None
    fallback_reason: str | None = None
    total_pressure: float = 0.0
    active_points: int = 0
    max_pressure: float = 0.0


@dataclass(frozen=True)
class PostureSegment:
    start_time: float
    end_time: float
    duration: float
    occupancy_state: str
    posture: str | None
    mean_confidence: float | None
    min_confidence: float | None
    boundary_ratio: float
    second_label: str | None


def load_csv_playback(path: Path | str, fallback_fps: float = 20.0) -> CsvPlaybackData:
    source = Path(path)
    try:
        timestamps, frames = read_sensor_csv(source)
    except Exception as exc:
        raise CsvFormatError(f"CSV格式错误，无法解析为FlexPressureVision 16x16帧: {source.name}: {exc}") from exc
    if len(frames) == 0:
        raise CsvFormatError(f"CSV为空，未找到任何16x16压力帧: {source.name}")
    frame_times, fps, time_source = infer_frame_times(timestamps, len(frames), fallback_fps=fallback_fps)
    return CsvPlaybackData(
        path=source,
        timestamps=timestamps,
        frames=frames,
        frame_times_s=frame_times,
        fps=fps,
        time_source=time_source,
    )


def infer_frame_times(timestamps: list[str], frame_count: int, fallback_fps: float = 20.0) -> tuple[list[float], float, str]:
    if frame_count <= 0:
        return [], float(fallback_fps), "empty"
    parsed = [_parse_timestamp(item) for item in timestamps]
    if all(item is not None for item in parsed) and len(parsed) == frame_count and frame_count > 1:
        first = parsed[0]
        last = parsed[-1]
        assert first is not None and last is not None
        duration = (last - first).total_seconds()
        if duration > 0:
            times = np.linspace(0.0, duration, frame_count).tolist()
            fps = (frame_count - 1) / duration
            return [round(float(item), 6) for item in times], float(fps), "csv_timestamp"
    interval = 1.0 / max(float(fallback_fps), 1e-6)
    return [round(i * interval, 6) for i in range(frame_count)], float(fallback_fps), "fallback_fps"


def load_runtime_recognizer(
    model_version: str = "v1",
    model_path: Path | str | None = None,
    prototype_bank_path: Path | str | None = None,
) -> Recognizer:
    try:
        return Recognizer(model_version=model_version, model_path=model_path, prototype_bank_path=prototype_bank_path)
    except Exception as exc:
        raise CsvGuiError(f"模型加载失败: {exc}") from exc


class CsvRecognitionSession:
    def __init__(self, data: CsvPlaybackData, recognizer: object) -> None:
        self.data = data
        self.recognizer = recognizer
        self.index = 0
        self.playing = True
        self.predictions: list[FramePrediction] = []
        self._closed_segments: list[PostureSegment] = []
        self._segment_records: list[FramePrediction] = []
        self._segment_key: tuple[str, str | None, bool] | None = None

    def step(self) -> FramePrediction | None:
        if not self.playing:
            return None
        return self._process_next()

    def step_once(self) -> FramePrediction | None:
        was_playing = self.playing
        self.playing = True
        record = self._process_next()
        self.playing = was_playing
        return record

    def process_all(self) -> list[FramePrediction]:
        self.playing = True
        while self.index < self.data.frame_count:
            self._process_next()
        return list(self.predictions)

    def pause(self) -> None:
        self.playing = False

    def resume(self) -> None:
        self.playing = True

    def stop(self) -> None:
        self.reset()
        self.playing = False

    def reset(self) -> None:
        reset = getattr(self.recognizer, "reset", None)
        if callable(reset):
            reset()
        self.index = 0
        self.predictions = []
        self._closed_segments = []
        self._segment_records = []
        self._segment_key = None
        self.playing = True

    def seek(self, index: int) -> FramePrediction | None:
        if self.data.frame_count == 0:
            return None
        target = max(0, min(int(index), self.data.frame_count - 1))
        was_playing = self.playing
        self.reset()
        self.playing = True
        record = None
        while self.index <= target:
            record = self._process_next()
        self.playing = was_playing
        return record

    @property
    def segments(self) -> list[PostureSegment]:
        segments = list(self._closed_segments)
        if self._segment_records:
            segments.append(self._make_segment(self._segment_records, self._segment_end_time(self._segment_records[-1])))
        return segments

    def summary(self) -> dict[str, object]:
        model_info = model_export_info(self.recognizer)
        segments = self.segments
        status_durations: Counter[str] = Counter()
        posture_durations: Counter[str] = Counter()
        for segment in segments:
            status = segment.occupancy_state
            status_durations[status] += segment.duration
            if segment.posture:
                posture_durations[segment.posture] += segment.duration
        human_records = [record for record in self.predictions if record.occupancy_state == "HUMAN"]
        posture_records = [record for record in self.predictions if record.posture is not None]
        confidences = [record.posture_confidence for record in posture_records if record.posture_confidence is not None]
        first_human = human_records[0].timestamp if human_records else None
        first_posture = posture_records[0].timestamp if posture_records else None
        labels = [record.posture for record in posture_records if record.posture]
        switches = sum(1 for prev, curr in zip(labels, labels[1:]) if prev != curr)
        object_posture = [
            record.frame_index
            for record in self.predictions
            if record.occupancy_state != "HUMAN" and record.posture is not None
        ]
        boundary_count = sum(1 for record in self.predictions if record.is_boundary)
        main_posture = posture_durations.most_common(1)[0][0] if posture_durations else None
        return {
            **model_info,
            "file_name": self.data.path.name,
            "frame_count": self.data.frame_count,
            "processed_frames": len(self.predictions),
            "export_complete": len(self.predictions) >= self.data.frame_count,
            "processed_duration_s": round(self.predictions[-1].timestamp, 4) if self.predictions else 0.0,
            "total_duration_s": round(self.data.duration_s, 4),
            "empty_time_s": round(float(status_durations.get("EMPTY", 0.0)), 4),
            "human_time_s": round(float(status_durations.get("HUMAN", 0.0)), 4),
            "object_unknown_time_s": round(float(status_durations.get("OBJECT", 0.0) + status_durations.get("UNKNOWN", 0.0)), 4),
            "first_posture_delay_s": None
            if first_human is None or first_posture is None
            else round(float(first_posture - first_human), 4),
            "main_posture": main_posture,
            "posture_durations_s": {key: round(float(value), 4) for key, value in posture_durations.items()},
            "average_confidence": None if not confidences else round(float(np.mean(confidences)), 4),
            "boundary_ratio": round(boundary_count / max(len(self.predictions), 1), 4),
            "label_switch_count": switches,
            "object_entered_posture_model": bool(object_posture),
            "object_posture_frames": object_posture,
            "time_source": self.data.time_source,
            "fps": round(float(self.data.fps), 4),
        }

    def export_results(self, output_root: Path | str = Path("recognizer/gui_outputs")) -> dict[str, Path]:
        root = Path(output_root)
        model_info = model_export_info(self.recognizer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = root / f"{self.data.path.stem}_{timestamp}"
        target.mkdir(parents=True, exist_ok=True)
        frame_path = target / "frame_predictions.csv"
        segment_path = target / "posture_segments.csv"
        summary_path = target / "summary.json"

        frame_fields = [
            "model_version",
            "promotion_status",
            "promotion_evidence",
            "promotion_manifest_hash",
            "model_artifact_sha256",
            "metadata_sha256",
            "runtime_config_sha256",
            "submodel_sha256",
            "subprototype_bank_sha256",
            "subruntime_config_sha256",
            "lateral_submodel_sha256",
            "lateral_prototype_bank_sha256",
            "lateral_runtime_config_sha256",
            "model_bundle_sha256",
            "timestamp",
            "frame_index",
            "occupancy_state",
            "occupancy_confidence",
            "seat_state",
            "display_status",
            "posture",
            "posture_confidence",
            "raw_label",
            "raw_confidence",
            "second_label",
            "margin",
            "is_boundary",
            "boundary_reason",
            "prototype_diagnosis",
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
            "final_priority_branch",
            "selected_branch",
            "override_reason",
            "fallback_reason",
            "total_pressure",
            "active_points",
            "max_pressure",
        ]
        with frame_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=frame_fields)
            writer.writeheader()
            for record in self.predictions:
                row = asdict(record)
                row.update(model_info)
                writer.writerow({key: row.get(key) for key in frame_fields})

        segment_fields = [
            "model_version",
            "promotion_status",
            "promotion_evidence",
            "promotion_manifest_hash",
            "model_artifact_sha256",
            "metadata_sha256",
            "runtime_config_sha256",
            "submodel_sha256",
            "subprototype_bank_sha256",
            "subruntime_config_sha256",
            "lateral_submodel_sha256",
            "lateral_prototype_bank_sha256",
            "lateral_runtime_config_sha256",
            "model_bundle_sha256",
            "start_time",
            "end_time",
            "duration",
            "occupancy_state",
            "posture",
            "mean_confidence",
            "boundary_ratio",
        ]
        with segment_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=segment_fields)
            writer.writeheader()
            for segment in self.segments:
                row = asdict(segment)
                row.update(model_info)
                writer.writerow({key: row.get(key) for key in segment_fields})
        summary_path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "directory": target,
            "frame_predictions": frame_path,
            "posture_segments": segment_path,
            "summary": summary_path,
        }

    def _process_next(self) -> FramePrediction | None:
        if self.index >= self.data.frame_count:
            return None
        frame = as_frame(self.data.frames[self.index])
        result = self.recognizer.predict(frame)
        record = frame_record_from_result(
            frame=frame,
            result=result,
            frame_index=self.index,
            timestamp=self.data.frame_times_s[self.index],
        )
        self.predictions.append(record)
        self._update_segments(record)
        self.index += 1
        return record

    def _update_segments(self, record: FramePrediction) -> None:
        key = segment_key(record)
        if self._segment_key is None:
            self._segment_key = key
            self._segment_records = [record]
            return
        if key != self._segment_key:
            previous = self._segment_records[-1]
            self._closed_segments.append(self._make_segment(self._segment_records, record.timestamp))
            self._segment_key = key
            self._segment_records = [record]
            return
        self._segment_records.append(record)

    def _segment_end_time(self, last_record: FramePrediction) -> float:
        return last_record.timestamp + self.data.frame_interval_s(last_record.frame_index)

    @staticmethod
    def _make_segment(records: list[FramePrediction], end_time: float) -> PostureSegment:
        start = records[0].timestamp
        duration = max(0.0, end_time - start)
        confidences = [record.posture_confidence for record in records if record.posture_confidence is not None]
        second_labels = [record.second_label for record in records if record.second_label]
        return PostureSegment(
            start_time=round(float(start), 4),
            end_time=round(float(end_time), 4),
            duration=round(float(duration), 4),
            occupancy_state=records[0].occupancy_state,
            posture=records[0].posture,
            mean_confidence=None if not confidences else round(float(np.mean(confidences)), 4),
            min_confidence=None if not confidences else round(float(np.min(confidences)), 4),
            boundary_ratio=round(sum(1 for record in records if record.is_boundary) / max(len(records), 1), 4),
            second_label=Counter(second_labels).most_common(1)[0][0] if second_labels else None,
        )


def frame_record_from_result(
    frame: np.ndarray,
    result: dict[str, Any],
    frame_index: int,
    timestamp: float,
) -> FramePrediction:
    occupancy = str(result.get("occupancy") or result.get("occupancy_state") or "UNKNOWN")
    seat_state = str(result.get("seat_state") or occupancy)
    posture = result.get("posture") if occupancy == "HUMAN" else None
    is_human = occupancy == "HUMAN"
    features = result.get("occupancy_features") if isinstance(result.get("occupancy_features"), dict) else {}
    total = float(features.get("total_pressure", float(np.asarray(frame).sum())))
    active_points = int(features.get("active_points", int((np.asarray(frame) > 15.0).sum())))
    return FramePrediction(
        timestamp=round(float(timestamp), 6),
        frame_index=int(frame_index),
        occupancy_state=occupancy,
        occupancy_confidence=_optional_float(result.get("occupancy_confidence")),
        seat_state=seat_state,
        display_status=display_status(result),
        posture=str(posture) if posture is not None else None,
        posture_confidence=_optional_float(result.get("posture_confidence")) if is_human else None,
        raw_label=str(result.get("raw_label")) if is_human and result.get("raw_label") else None,
        raw_confidence=_optional_float(result.get("raw_confidence")) if is_human else None,
        second_label=str(result.get("second_label")) if is_human and result.get("second_label") else None,
        margin=_optional_float(result.get("margin")) if is_human else None,
        is_boundary=bool(result.get("is_boundary", False)),
        boundary_reason=str(result.get("boundary_reason")) if is_human and result.get("boundary_reason") else None,
        prototype_diagnosis=_summarize_prototype_diagnosis(result.get("prototype_diagnosis")) if is_human else None,
        parent_posture_label=str(result.get("parent_posture_label")) if is_human and result.get("parent_posture_label") else None,
        fine_posture_label=str(result.get("fine_posture_label")) if is_human and result.get("fine_posture_label") else None,
        final_display_label=str(result.get("final_display_label")) if is_human and result.get("final_display_label") else None,
        subclassifier_triggered=bool(result.get("subclassifier_triggered")) if is_human else None,
        subclassifier_gate_reason=str(result.get("subclassifier_gate_reason"))
        if is_human and result.get("subclassifier_gate_reason")
        else None,
        fine_confidence=_optional_float(result.get("fine_confidence")) if is_human else None,
        fine_margin=_optional_float(result.get("fine_margin")) if is_human else None,
        fine_boundary=bool(result.get("fine_boundary")) if is_human and result.get("fine_boundary") is not None else None,
        fine_boundary_reasons=_summarize_reasons(result.get("fine_boundary_reasons")) if is_human else None,
        fine_prototype_label=str(result.get("fine_prototype_label")) if is_human and result.get("fine_prototype_label") else None,
        fine_prototype_distance=_optional_float(result.get("fine_prototype_distance")) if is_human else None,
        fallback_used=bool(result.get("fallback_used")) if is_human and result.get("fallback_used") is not None else None,
        parent_model_version=str(result.get("parent_model_version")) if is_human and result.get("parent_model_version") else None,
        submodel_version=str(result.get("submodel_version")) if is_human and result.get("submodel_version") else None,
        lateral_subclassifier_triggered=bool(result.get("lateral_subclassifier_triggered"))
        if is_human and result.get("lateral_subclassifier_triggered") is not None
        else None,
        lateral_gate_reason=str(result.get("lateral_gate_reason")) if is_human and result.get("lateral_gate_reason") else None,
        lateral_posture_label=str(result.get("lateral_posture_label")) if is_human and result.get("lateral_posture_label") else None,
        lateral_confidence=_optional_float(result.get("lateral_confidence")) if is_human else None,
        lateral_margin=_optional_float(result.get("lateral_margin")) if is_human else None,
        lateral_boundary=bool(result.get("lateral_boundary")) if is_human and result.get("lateral_boundary") is not None else None,
        lateral_boundary_reasons=_summarize_reasons(result.get("lateral_boundary_reasons")) if is_human else None,
        lateral_prototype_label=str(result.get("lateral_prototype_label")) if is_human and result.get("lateral_prototype_label") else None,
        lateral_prototype_distance=_optional_float(result.get("lateral_prototype_distance")) if is_human else None,
        lateral_fallback_used=bool(result.get("lateral_fallback_used")) if is_human and result.get("lateral_fallback_used") is not None else None,
        lateral_submodel_version=str(result.get("lateral_submodel_version")) if is_human and result.get("lateral_submodel_version") else None,
        lateral_second_label=str(result.get("lateral_second_label")) if is_human and result.get("lateral_second_label") else None,
        lateral_second_distance=_optional_float(result.get("lateral_second_distance")) if is_human else None,
        lateral_prototype_margin=_optional_float(result.get("lateral_prototype_margin")) if is_human else None,
        lateral_out_of_distribution=bool(result.get("lateral_out_of_distribution"))
        if is_human and result.get("lateral_out_of_distribution") is not None
        else None,
        lateral_temporal_state=str(result.get("lateral_temporal_state")) if is_human and result.get("lateral_temporal_state") else None,
        lateral_stable_label=str(result.get("lateral_stable_label")) if is_human and result.get("lateral_stable_label") else None,
        lateral_fallback_requested=bool(result.get("lateral_fallback_requested")) if is_human and result.get("lateral_fallback_requested") is not None else None,
        final_priority_branch=str(result.get("final_priority_branch")) if is_human and result.get("final_priority_branch") else None,
        selected_branch=str(result.get("selected_branch")) if is_human and result.get("selected_branch") else None,
        override_reason=str(result.get("override_reason")) if is_human and result.get("override_reason") else None,
        fallback_reason=str(result.get("fallback_reason")) if is_human and result.get("fallback_reason") else None,
        total_pressure=round(total, 4),
        active_points=active_points,
        max_pressure=round(float(np.asarray(frame).max()) if np.asarray(frame).size else 0.0, 4),
    )


def sha256_file(path: Path | str | None) -> str | None:
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


def model_export_info(recognizer: object) -> dict[str, object]:
    artifact_identity = getattr(recognizer, "artifact_identity", None)
    if callable(artifact_identity):
        info = artifact_identity()
        payload = {
            "model_version": info.get("model_version"),
            "model_path": info.get("model_path"),
            "metadata_path": info.get("metadata_path"),
            "prototype_bank_path": info.get("prototype_bank_path"),
            "runtime_config_path": info.get("runtime_config_path"),
            "submodel_path": info.get("submodel_path"),
            "subprototype_bank_path": info.get("subprototype_bank_path"),
            "subruntime_config_path": info.get("subruntime_config_path"),
            "lateral_submodel_path": info.get("lateral_submodel_path"),
            "lateral_prototype_bank_path": info.get("lateral_prototype_bank_path"),
            "lateral_runtime_config_path": info.get("lateral_runtime_config_path"),
            "model_bundle_path": info.get("model_bundle_path"),
            "model_artifact_sha256": info.get("model_artifact_sha256"),
            "metadata_sha256": info.get("metadata_sha256"),
            "prototype_bank_sha256": info.get("prototype_bank_sha256"),
            "runtime_config_sha256": info.get("runtime_config_sha256"),
            "submodel_sha256": info.get("submodel_sha256"),
            "subprototype_bank_sha256": info.get("subprototype_bank_sha256"),
            "subruntime_config_sha256": info.get("subruntime_config_sha256"),
            "lateral_submodel_sha256": info.get("lateral_submodel_sha256"),
            "lateral_prototype_bank_sha256": info.get("lateral_prototype_bank_sha256"),
            "lateral_runtime_config_sha256": info.get("lateral_runtime_config_sha256"),
            "model_bundle_sha256": info.get("model_bundle_sha256"),
        }
        payload.update(promotion_export_info(payload["model_version"]))
        return payload

    model_path = getattr(recognizer, "model_path", None)
    metadata_path = getattr(recognizer, "metadata_path", None)
    prototype_bank_path = getattr(recognizer, "prototype_bank_path", None)
    runtime_config_path = getattr(recognizer, "runtime_config_path", None)
    payload = {
        "model_version": getattr(recognizer, "model_version", "unknown"),
        "model_path": None if model_path is None else str(model_path),
        "metadata_path": None if metadata_path is None else str(metadata_path),
        "prototype_bank_path": None if prototype_bank_path is None else str(prototype_bank_path),
        "runtime_config_path": None if runtime_config_path is None else str(runtime_config_path),
        "model_artifact_sha256": sha256_file(model_path),
        "metadata_sha256": sha256_file(metadata_path),
        "prototype_bank_sha256": sha256_file(prototype_bank_path),
        "runtime_config_sha256": sha256_file(runtime_config_path),
        "submodel_sha256": None,
        "subprototype_bank_sha256": None,
        "subruntime_config_sha256": None,
        "lateral_submodel_sha256": None,
        "lateral_prototype_bank_sha256": None,
        "lateral_runtime_config_sha256": None,
        "model_bundle_sha256": None,
    }
    payload.update(promotion_export_info(payload["model_version"]))
    return payload


def promotion_export_info(model_version: object) -> dict[str, object]:
    if model_version != "v2_2_candidate":
        return {
            "promotion_status": None,
            "promotion_evidence": None,
            "promotion_manifest_hash": None,
        }
    manifest = Path(__file__).resolve().parent / "models" / "v2_2_promotion_manifest.json"
    status = None
    evidence = None
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            status = payload.get("promotion_status")
            evidence = payload.get("promotion_reason")
        except Exception:
            status = "promotion_manifest_unreadable"
            evidence = None
    return {
        "promotion_status": status,
        "promotion_evidence": evidence,
        "promotion_manifest_hash": sha256_file(manifest),
    }


def display_status(result: dict[str, Any]) -> str:
    if result.get("posture") is not None:
        return "POSTURE"
    seat_state = str(result.get("seat_state") or result.get("occupancy") or "UNKNOWN")
    if seat_state == "HUMAN_RECOGNIZING":
        return "POSTURE"
    return seat_state


def segment_key(record: FramePrediction) -> tuple[str, str | None, bool]:
    boundary_posture = "边界姿势/低置信度" if record.display_status == "POSTURE" and record.posture is None else None
    return (record.display_status, record.posture or boundary_posture, bool(record.is_boundary))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), "%Y/%m/%d_%H:%M:%S")
    except ValueError:
        return None


def _summarize_prototype_diagnosis(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        label = value.get("label")
        matched = value.get("matched_prototype_id")
        agrees = value.get("agrees_with_rf")
        parts = []
        if label is not None:
            parts.append(f"proto={label}")
        if matched is not None:
            parts.append(f"id={matched}")
        if agrees is not None:
            parts.append(f"agrees={agrees}")
        return "; ".join(parts) if parts else json.dumps(value, ensure_ascii=False)
    return str(value)


def _summarize_reasons(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)
    return str(value)
