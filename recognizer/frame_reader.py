from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable

import numpy as np

from .data_loader import read_sensor_csv
from .feature_extractor import FRAME_SHAPE, as_frame
from .serial_protocol import ParsedPressureFrame, PressurePacketParser


DEFAULT_SERIAL_BAUDRATE = 460800
DEFAULT_SERIAL_TIMEOUT = 0.05
DEFAULT_SERIAL_READ_SIZE = 1024


@dataclass(frozen=True)
class RecordingBoundaryResult:
    buffered_bytes_cleared: int
    queued_frames_cleared: int


@dataclass
class _ReaderControlRequest:
    action: str
    raw_chunk_listener: Callable[[bytes], None] | None = None
    parsed_frame_listener: Callable[[ParsedPressureFrame], None] | None = None
    clear_queue: bool = False
    done: threading.Event | None = None
    result: RecordingBoundaryResult | None = None
    error: BaseException | None = None


def _load_serial_comports() -> Callable[[], list[Any]]:
    try:
        from serial.tools import list_ports
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyserial is required for serial port scanning. Install requirements-macos.txt.") from exc
    return list_ports.comports


def list_serial_ports() -> list[Any]:
    return list(_load_serial_comports()())


class FrameReader:
    def read_frame(self) -> np.ndarray:
        raise NotImplementedError


class CSVReplayReader(FrameReader):
    def __init__(self, path: Path | str, loop: bool = False, orientation: str = "normal") -> None:
        self.path = Path(path)
        _, self.frames = read_sensor_csv(self.path)
        self.loop = bool(loop)
        self.orientation = orientation
        self.index = 0

    def read_frame(self) -> np.ndarray:
        if len(self.frames) == 0:
            raise EOFError(f"No frames in {self.path}")
        if self.index >= len(self.frames):
            if not self.loop:
                raise EOFError(str(self.path))
            self.index = 0
        frame = self.frames[self.index]
        self.index += 1
        return apply_orientation(frame, self.orientation)


@dataclass
class SerialFrameReader(FrameReader):
    port: str
    baudrate: int = DEFAULT_SERIAL_BAUDRATE
    rows: int = 16
    cols: int = 16
    delimiter: str = ","
    timestamp_enabled: bool = False
    frame_header: str | None = None
    frame_footer: str | None = None
    orientation: str = "normal"
    timeout: float = 1.0
    serial_timeout: float = DEFAULT_SERIAL_TIMEOUT
    queue_size: int = 3
    read_size: int = DEFAULT_SERIAL_READ_SIZE
    serial_factory: Callable[..., Any] | None = None
    raw_chunk_listener: Callable[[bytes], None] | None = None
    parsed_frame_listener: Callable[[ParsedPressureFrame], None] | None = None

    parser: PressurePacketParser = field(init=False)
    received_bytes: int = field(init=False, default=0)
    valid_frames: int = field(init=False, default=0)
    dropped_queue_frames: int = field(init=False, default=0)
    last_error: BaseException | None = field(init=False, default=None)
    _queue: Queue[np.ndarray] = field(init=False)
    _stop_event: threading.Event = field(init=False)
    _thread: threading.Thread | None = field(init=False, default=None)
    _serial: Any | None = field(init=False, default=None)
    _frame_timestamps: deque[float] = field(init=False)
    _stats_lock: threading.Lock = field(init=False)
    _control_queue: Queue[_ReaderControlRequest] = field(init=False)

    def __post_init__(self) -> None:
        self.parser = PressurePacketParser()
        self._queue = Queue(maxsize=max(1, int(self.queue_size)))
        self._stop_event = threading.Event()
        self._frame_timestamps = deque()
        self._stats_lock = threading.Lock()
        self._control_queue = Queue()

    @staticmethod
    def list_ports() -> list[Any]:
        return list_serial_ports()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def invalid_frames(self) -> int:
        return self.parser.invalid_packets

    @property
    def discarded_bytes(self) -> int:
        return self.parser.discarded_bytes

    @property
    def current_fps(self) -> float:
        with self._stats_lock:
            return self._current_fps_locked()

    def start(self) -> None:
        if self.is_running:
            return
        self._reset_runtime_state()
        self._stop_event.clear()
        try:
            self._serial = self._open_serial()
        except BaseException as exc:
            self.last_error = exc
            raise
        self._thread = threading.Thread(target=self._read_loop, name=f"SerialFrameReader-{self.port}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except BaseException as exc:
                self.last_error = exc
            finally:
                self._serial = None

    def read_frame(self) -> np.ndarray:
        if not self.is_running:
            self.start()
        try:
            frame = self._queue.get(timeout=self.timeout)
        except Empty as exc:
            raise TimeoutError(f"No serial pressure frame received from {self.port}") from exc
        oriented = apply_orientation(frame, self.orientation)
        return np.asarray(oriented, dtype=np.float32)

    def begin_recording_boundary(
        self,
        *,
        raw_chunk_listener: Callable[[bytes], None],
        parsed_frame_listener: Callable[[ParsedPressureFrame], None],
        clear_queue: bool = True,
        timeout: float = 1.0,
    ) -> RecordingBoundaryResult:
        request = _ReaderControlRequest(
            action="begin_recording_boundary",
            raw_chunk_listener=raw_chunk_listener,
            parsed_frame_listener=parsed_frame_listener,
            clear_queue=clear_queue,
        )
        result = self._submit_control_request(request, timeout=timeout)
        assert isinstance(result, RecordingBoundaryResult)
        return result

    def end_recording_boundary(self, *, timeout: float = 1.0) -> None:
        request = _ReaderControlRequest(action="end_recording_boundary")
        self._submit_control_request(request, timeout=timeout)

    def stats(self) -> dict[str, object]:
        return {
            "received_bytes": self.received_bytes,
            "valid_frames": self.valid_frames,
            "invalid_frames": self.invalid_frames,
            "discarded_bytes": self.discarded_bytes,
            "dropped_queue_frames": self.dropped_queue_frames,
            "current_fps": self.current_fps,
            "last_error": self.last_error,
        }

    def _reset_runtime_state(self) -> None:
        self.parser.reset()
        self._queue = Queue(maxsize=max(1, int(self.queue_size)))
        self._control_queue = Queue()
        self.received_bytes = 0
        self.valid_frames = 0
        self.dropped_queue_frames = 0
        self.last_error = None
        with self._stats_lock:
            self._frame_timestamps.clear()

    def _open_serial(self) -> Any:
        serial_kwargs = {
            "port": self.port,
            "baudrate": self.baudrate,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": self.serial_timeout,
            "xonxoff": False,
            "rtscts": False,
            "dsrdtr": False,
        }
        if self.serial_factory is not None:
            return self.serial_factory(**serial_kwargs)
        try:
            import serial
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyserial is required for SerialFrameReader. Install requirements-macos.txt.") from exc
        return serial.Serial(**serial_kwargs)

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            self._process_control_requests()
            try:
                chunk = self._serial.read(self.read_size) if self._serial is not None else b""
            except BaseException as exc:
                self.last_error = exc
                break
            if not chunk:
                continue
            chunk_bytes = bytes(chunk)
            self.received_bytes += len(chunk_bytes)
            self._notify_raw_chunk(chunk_bytes)
            try:
                parsed_frames = self.parser.feed(chunk_bytes)
            except BaseException as exc:
                self.last_error = exc
                break
            for parsed in parsed_frames:
                self._notify_parsed_frame(parsed)
                self._enqueue_frame(parsed.matrix)
                self.valid_frames += 1
                self._record_frame_timestamp()
        self._process_control_requests()

    def _submit_control_request(
        self,
        request: _ReaderControlRequest,
        *,
        timeout: float,
    ) -> RecordingBoundaryResult | None:
        request.done = threading.Event()
        if not self.is_running:
            self._execute_control_request(request)
            if request.error is not None:
                raise request.error
            return request.result
        self._control_queue.put(request)
        if not request.done.wait(timeout=timeout):
            raise TimeoutError(f"SerialFrameReader control request timed out: {request.action}")
        if request.error is not None:
            raise request.error
        return request.result

    def _process_control_requests(self) -> None:
        while True:
            try:
                request = self._control_queue.get_nowait()
            except Empty:
                return
            try:
                self._execute_control_request(request)
            except BaseException as exc:
                request.error = exc
                self.last_error = exc
            finally:
                if request.done is not None:
                    request.done.set()

    def _execute_control_request(self, request: _ReaderControlRequest) -> None:
        if request.action == "begin_recording_boundary":
            buffered_bytes_cleared = self.parser.clear_buffered_bytes()
            queued_frames_cleared = self._clear_frame_queue() if request.clear_queue else 0
            self.raw_chunk_listener = request.raw_chunk_listener
            self.parsed_frame_listener = request.parsed_frame_listener
            request.result = RecordingBoundaryResult(
                buffered_bytes_cleared=buffered_bytes_cleared,
                queued_frames_cleared=queued_frames_cleared,
            )
            return
        if request.action == "end_recording_boundary":
            self.raw_chunk_listener = None
            self.parsed_frame_listener = None
            request.result = RecordingBoundaryResult(buffered_bytes_cleared=0, queued_frames_cleared=0)
            return
        raise ValueError(f"Unknown SerialFrameReader control action: {request.action}")

    def _clear_frame_queue(self) -> int:
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except Empty:
                return cleared

    def _notify_raw_chunk(self, chunk: bytes) -> None:
        if self.raw_chunk_listener is None:
            return
        try:
            self.raw_chunk_listener(chunk)
        except BaseException as exc:
            self.last_error = exc

    def _notify_parsed_frame(self, parsed: ParsedPressureFrame) -> None:
        if self.parsed_frame_listener is None:
            return
        try:
            self.parsed_frame_listener(parsed)
        except BaseException as exc:
            self.last_error = exc

    def _enqueue_frame(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
            return
        except Full:
            pass
        try:
            self._queue.get_nowait()
            self.dropped_queue_frames += 1
        except Empty:
            pass
        try:
            self._queue.put_nowait(frame)
        except Full:
            self.dropped_queue_frames += 1

    def _record_frame_timestamp(self) -> None:
        with self._stats_lock:
            now = time.monotonic()
            self._frame_timestamps.append(now)
            self._prune_frame_timestamps_locked(now)

    def _current_fps_locked(self) -> float:
        now = time.monotonic()
        self._prune_frame_timestamps_locked(now)
        return float(len(self._frame_timestamps))

    def _prune_frame_timestamps_locked(self, now: float) -> None:
        cutoff = now - 1.0
        while self._frame_timestamps and self._frame_timestamps[0] < cutoff:
            self._frame_timestamps.popleft()


def apply_orientation(frame: np.ndarray, orientation: str = "normal") -> np.ndarray:
    arr = as_frame(frame)
    if orientation == "normal":
        return arr
    if orientation == "flip_lr":
        return np.flip(arr, axis=1)
    if orientation == "flip_ud":
        return np.flip(arr, axis=0)
    if orientation == "rot180":
        return np.rot90(arr, 2)
    if orientation == "transpose":
        return arr.T.reshape(FRAME_SHAPE)
    raise ValueError(f"Unknown orientation: {orientation}")
