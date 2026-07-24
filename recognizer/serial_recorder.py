from __future__ import annotations

import csv
import json
import platform
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

from .csv_gui_core import FramePrediction
from .frame_orientation import orientation_transform_name
from .serial_gui_core import SerialRecognitionResult, apply_sensor_and_orientation
from .serial_protocol import PACKET_SIZE, ParsedPressureFrame


SCHEMA_VERSION = "serial_capture_v1"
CSV_TIMESTAMP_FORMAT = "%Y/%m/%d_%H:%M:%S"
DIR_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
ILLEGAL_WINDOWS_FILENAME_CHARS = r'<>:"/\\|?*'
SERIAL_TEXT_FILENAME = "serial_raw_data.txt"
SERIAL_TEXT_FORMAT = "uppercase hexadecimal bytes, one valid 263-byte packet per line"
PRESSURE_FRAMES_FILENAME = "pressure_frames.csv"
PRESSURE_FRAMES_RAW_FILENAME = "pressure_frames_raw.csv"


@dataclass(frozen=True)
class _RecorderEvent:
    kind: str
    payload: Any


def sanitize_capture_label(label: str) -> str:
    cleaned = re.sub(f"[{re.escape(ILLEGAL_WINDOWS_FILENAME_CHARS)}]", "_", label.strip())
    cleaned = cleaned.rstrip(" .")
    return cleaned or "capture"


class SerialDataRecorder:
    def __init__(self, *, queue_size: int = 2048, clock: Any = time.monotonic) -> None:
        self.queue_size = max(1, int(queue_size))
        self.clock = clock
        self.capture_dir: Path | None = None
        self.last_error: BaseException | None = None

        self._queue: Queue[_RecorderEvent] = Queue(maxsize=self.queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._accepting = False

        self._raw_handle: Any | None = None
        self._frame_handle: Any | None = None
        self._raw_frame_handle: Any | None = None
        self._serial_text_handle: Any | None = None
        self._prediction_handle: Any | None = None
        self._prediction_writer: csv.DictWriter | None = None
        self._metadata: dict[str, Any] = {}
        self._start_monotonic: float | None = None

        self._raw_bytes_saved = 0
        self._valid_frames_saved = 0
        self._valid_packets_text_saved = 0
        self._predictions_saved = 0
        self._dropped_events = 0
        self._data_complete = True

    @property
    def is_recording(self) -> bool:
        return bool(self._accepting and self._thread is not None and self._thread.is_alive())

    def start(
        self,
        *,
        output_root: Path | str,
        label: str,
        trial: int,
        serial_port: str,
        baudrate: int,
        orientation: str,
        model_version: str,
        sensor_rotation_degrees: int = 0,
        serial_reader_stats_start: dict[str, Any] | None = None,
    ) -> Path:
        if self.is_recording:
            raise RuntimeError("SerialDataRecorder is already recording")
        self.stop()
        try:
            trial_int = int(trial)
            if trial_int <= 0:
                raise ValueError("trial must be a positive integer")
            sensor_rotation = int(sensor_rotation_degrees)
            transform_name = orientation_transform_name(sensor_rotation)
            root = Path(output_root)
            root.mkdir(parents=True, exist_ok=True)
            capture_dir = self._unique_capture_dir(root, sanitize_capture_label(label), trial_int)
            capture_dir.mkdir(parents=True, exist_ok=False)

            self.capture_dir = capture_dir
            self.last_error = None
            self._queue = Queue(maxsize=self.queue_size)
            self._stop_event.clear()
            self._accepting = True
            self._raw_bytes_saved = 0
            self._valid_frames_saved = 0
            self._valid_packets_text_saved = 0
            self._predictions_saved = 0
            self._dropped_events = 0
            self._data_complete = True
            self._start_monotonic = float(self.clock())

            self._raw_handle = (capture_dir / "raw_stream.bin").open("wb")
            self._frame_handle = (capture_dir / PRESSURE_FRAMES_FILENAME).open("w", encoding="ascii", newline="")
            self._raw_frame_handle = (capture_dir / PRESSURE_FRAMES_RAW_FILENAME).open("w", encoding="ascii", newline="")
            self._serial_text_handle = (capture_dir / SERIAL_TEXT_FILENAME).open("w", encoding="ascii", newline="")
            self._prediction_handle = (capture_dir / "recognition_results.csv").open("w", encoding="utf-8-sig", newline="")
            self._prediction_writer = csv.DictWriter(self._prediction_handle, fieldnames=_prediction_fieldnames())
            self._prediction_writer.writeheader()

            start_time = datetime.now()
            self._metadata = {
                "schema_version": SCHEMA_VERSION,
                "label": label.strip(),
                "trial": trial_int,
                "platform": platform.system(),
                "serial_port": str(serial_port),
                "baudrate": int(baudrate),
                "orientation": str(orientation),
                "sensor_rotation_degrees": sensor_rotation,
                "orientation_transform": transform_name,
                "physical_frame_saved": True,
                "canonical_frame_saved": True,
                "canonical_orientation": "legacy_training_orientation",
                "model_version": str(model_version),
                "start_time": start_time.isoformat(timespec="microseconds"),
                "end_time": None,
                "duration_s": 0.0,
                "raw_bytes_saved": 0,
                "valid_frames_saved": 0,
                "valid_packets_text_saved": 0,
                "predictions_saved": 0,
                "serial_text_filename": SERIAL_TEXT_FILENAME,
                "serial_text_format": SERIAL_TEXT_FORMAT,
                "serial_text_transform": "none",
                "pressure_frames_filename": PRESSURE_FRAMES_FILENAME,
                "pressure_frames_format": "canonical model-facing 16x16 pressure matrix",
                "pressure_frames_raw_filename": PRESSURE_FRAMES_RAW_FILENAME,
                "pressure_frames_raw_format": "raw physical 16x16 pressure matrix from serial protocol parser",
                "raw_stream_transform": "none",
                "checksum_validation_enabled": False,
                "packet_size": PACKET_SIZE,
                "matrix_shape": [16, 16],
                "capture_completed": False,
                "data_complete": True,
                "recorder_dropped_events": 0,
                "recorder_error": None,
                "serial_reader_stats_start": serial_reader_stats_start or {},
                "serial_reader_stats_end": None,
                "csv_timestamp_format": "YYYY/MM/DD_HH:MM:SS",
                "high_precision_frame_time_source": "metadata_and_recognition_results",
            }
            self._write_metadata()
            self._thread = threading.Thread(target=self._write_loop, name="SerialDataRecorder", daemon=True)
            self._thread.start()
            return capture_dir
        except BaseException as exc:
            self.last_error = exc
            self._accepting = False
            self._close_files()
            raise

    def stop(self, *, serial_reader_stats_end: dict[str, Any] | None = None, timeout: float = 3.0) -> None:
        self._accepting = False
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._thread is not None and self._thread.is_alive():
            self._data_complete = False
        self._flush_files()
        self._close_files()
        if self.capture_dir is not None and self._metadata:
            now = datetime.now()
            duration = 0.0 if self._start_monotonic is None else max(0.0, float(self.clock() - self._start_monotonic))
            self._metadata.update(
                {
                    "end_time": now.isoformat(timespec="microseconds"),
                    "duration_s": round(duration, 6),
                    "raw_bytes_saved": self._raw_bytes_saved,
                    "valid_frames_saved": self._valid_frames_saved,
                    "valid_packets_text_saved": self._valid_packets_text_saved,
                    "predictions_saved": self._predictions_saved,
                    "capture_completed": self.last_error is None and self._data_complete and not (self._thread and self._thread.is_alive()),
                    "data_complete": self.last_error is None and self._data_complete,
                    "recorder_dropped_events": self._dropped_events,
                    "recorder_error": None if self.last_error is None else str(self.last_error),
                    "serial_reader_stats_end": serial_reader_stats_end or self._metadata.get("serial_reader_stats_end"),
                }
            )
            try:
                self._write_metadata()
            except BaseException as exc:
                self.last_error = exc
        self._thread = None

    def record_raw_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._enqueue("raw", bytes(chunk))

    def record_parsed_frame(self, parsed_frame: ParsedPressureFrame | np.ndarray) -> None:
        raw_packet = parsed_frame.raw_packet if isinstance(parsed_frame, ParsedPressureFrame) else None
        matrix = parsed_frame.matrix if isinstance(parsed_frame, ParsedPressureFrame) else parsed_frame
        physical_frame = np.asarray(matrix, dtype=np.float32)
        sensor_rotation = int(self._metadata.get("sensor_rotation_degrees", 0))
        # The protocol parser produces a raw physical frame. The shared transform keeps
        # CSV output aligned with the live heatmap and Recognizer input.
        frame = apply_sensor_and_orientation(physical_frame, sensor_rotation, str(self._metadata.get("orientation", "原始")))
        self._enqueue("frame", (frame, raw_packet, np.ascontiguousarray(physical_frame, dtype=np.float32)))

    def record_prediction(self, result: SerialRecognitionResult | FramePrediction) -> None:
        if isinstance(result, SerialRecognitionResult):
            event_payload = result
        else:
            event_payload = SerialRecognitionResult(frame=np.zeros((16, 16), dtype=np.float32), prediction=result, raw_result={}, inference_ms=0.0)
        self._enqueue("prediction", event_payload)

    def stats(self) -> dict[str, Any]:
        duration = 0.0
        if self._start_monotonic is not None:
            duration = max(0.0, float(self.clock() - self._start_monotonic))
        return {
            "is_recording": self.is_recording,
            "capture_dir": self.capture_dir,
            "duration_s": duration,
            "raw_bytes_saved": self._raw_bytes_saved,
            "valid_frames_saved": self._valid_frames_saved,
            "valid_packets_text_saved": self._valid_packets_text_saved,
            "predictions_saved": self._predictions_saved,
            "recorder_queue_size": self._queue.qsize(),
            "recorder_dropped_events": self._dropped_events,
            "last_error": self.last_error,
            "data_complete": self._data_complete and self.last_error is None,
        }

    def _enqueue(self, kind: str, payload: Any) -> None:
        if not self._accepting:
            return
        try:
            self._queue.put_nowait(_RecorderEvent(kind=kind, payload=payload))
        except Full:
            self._dropped_events += 1
            self._data_complete = False

    def _write_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                self._write_event(event)
            except BaseException as exc:
                self.last_error = exc
                self._data_complete = False
            finally:
                self._queue.task_done()

    def _write_event(self, event: _RecorderEvent) -> None:
        if event.kind == "raw":
            assert self._raw_handle is not None
            self._raw_handle.write(event.payload)
            self._raw_bytes_saved += len(event.payload)
            return
        if event.kind == "frame":
            assert self._frame_handle is not None
            assert self._raw_frame_handle is not None
            frame, raw_packet, physical_frame = event.payload
            self._write_pressure_frame(frame)
            self._write_pressure_frame(physical_frame, handle=self._raw_frame_handle)
            self._valid_frames_saved += 1
            if raw_packet is not None and len(raw_packet) == PACKET_SIZE:
                self._write_serial_text_packet(raw_packet)
                self._valid_packets_text_saved += 1
            return
        if event.kind == "prediction":
            assert self._prediction_writer is not None
            self._prediction_writer.writerow(_prediction_row(event.payload))
            self._predictions_saved += 1
            return
        raise ValueError(f"Unknown recorder event: {event.kind}")

    def _write_pressure_frame(self, frame: np.ndarray, *, handle: Any | None = None) -> None:
        target = handle or self._frame_handle
        assert target is not None
        target.write(datetime.now().strftime(CSV_TIMESTAMP_FORMAT) + "\n")
        for row in np.asarray(frame, dtype=np.float32):
            target.write(",".join(f"{float(value):.10g}" for value in row) + "\n")

    def _write_serial_text_packet(self, raw_packet: bytes) -> None:
        assert self._serial_text_handle is not None
        self._serial_text_handle.write(" ".join(f"{byte:02X}" for byte in raw_packet) + "\n")

    def _unique_capture_dir(self, root: Path, label: str, trial: int) -> Path:
        base = root / f"{label}_{trial}_{datetime.now().strftime(DIR_TIMESTAMP_FORMAT)}"
        if not base.exists():
            return base
        suffix = 2
        while True:
            candidate = root / f"{base.name}_{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def _write_metadata(self) -> None:
        assert self.capture_dir is not None
        (self.capture_dir / "metadata.json").write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _flush_files(self) -> None:
        for handle in (self._raw_handle, self._frame_handle, self._raw_frame_handle, self._serial_text_handle, self._prediction_handle):
            if handle is None:
                continue
            try:
                handle.flush()
            except BaseException as exc:
                self.last_error = exc

    def _close_files(self) -> None:
        for attr in ("_raw_handle", "_frame_handle", "_raw_frame_handle", "_serial_text_handle", "_prediction_handle"):
            handle = getattr(self, attr)
            if handle is None:
                continue
            try:
                handle.close()
            except BaseException as exc:
                self.last_error = exc
            setattr(self, attr, None)
        self._prediction_writer = None


def _prediction_fieldnames() -> list[str]:
    return [
        "relative_time_s",
        "frame_index",
        "occupancy_state",
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
        "total_pressure",
        "max_pressure",
        "active_points",
        "inference_ms",
    ]


def _prediction_row(result: SerialRecognitionResult) -> dict[str, Any]:
    record = result.prediction
    if not isinstance(record, FramePrediction):
        record = FramePrediction(**asdict(record))
    return {
        "relative_time_s": _format_float(record.timestamp),
        "frame_index": record.frame_index,
        "occupancy_state": record.occupancy_state,
        "display_status": record.display_status,
        "posture": record.posture,
        "posture_confidence": _format_optional(record.posture_confidence),
        "raw_label": record.raw_label,
        "raw_confidence": _format_optional(record.raw_confidence),
        "second_label": record.second_label,
        "margin": _format_optional(record.margin),
        "is_boundary": record.is_boundary,
        "boundary_reason": record.boundary_reason,
        "prototype_diagnosis": record.prototype_diagnosis,
        "total_pressure": _format_float(record.total_pressure),
        "max_pressure": _format_float(record.max_pressure),
        "active_points": record.active_points,
        "inference_ms": _format_float(result.inference_ms),
    }


def _format_optional(value: float | None) -> str:
    return "" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    return f"{float(value):.6g}"
