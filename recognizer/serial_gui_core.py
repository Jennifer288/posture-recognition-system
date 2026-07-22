from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable

import numpy as np

from .csv_gui_core import FramePrediction, frame_record_from_result
from .feature_extractor import FRAME_SHAPE, as_frame


ORIENTATION_MODES = (
    "原始",
    "上下翻转",
    "左右翻转",
    "上下及左右翻转",
    "转置",
    "转置后上下翻转",
    "转置后左右翻转",
    "转置后上下及左右翻转",
)


@dataclass(frozen=True)
class SerialRecognitionResult:
    frame: np.ndarray
    prediction: FramePrediction
    raw_result: dict[str, Any]
    inference_ms: float


@dataclass
class _ControlRequest:
    action: str
    frame: np.ndarray | None = None
    done: threading.Event | None = None
    error: BaseException | None = None


def apply_orientation(frame: np.ndarray, mode: str) -> np.ndarray:
    arr = as_frame(frame)
    if mode == "原始":
        transformed = arr
    elif mode == "上下翻转":
        transformed = np.flipud(arr)
    elif mode == "左右翻转":
        transformed = np.fliplr(arr)
    elif mode == "上下及左右翻转":
        transformed = np.flipud(np.fliplr(arr))
    elif mode == "转置":
        transformed = arr.T
    elif mode == "转置后上下翻转":
        transformed = np.flipud(arr.T)
    elif mode == "转置后左右翻转":
        transformed = np.fliplr(arr.T)
    elif mode == "转置后上下及左右翻转":
        transformed = np.flipud(np.fliplr(arr.T))
    else:
        raise ValueError(f"未知方向设置: {mode}")
    return np.ascontiguousarray(transformed, dtype=np.float32)


class RecognitionWorker:
    def __init__(
        self,
        *,
        frame_source: Any,
        recognizer: Any,
        result_queue: Queue[SerialRecognitionResult] | None = None,
        orientation_mode: str | Callable[[], str] = "原始",
        connection_start_time: float | None = None,
        poll_timeout: float = 0.05,
        result_queue_size: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.frame_source = frame_source
        self.recognizer = recognizer
        self.result_queue: Queue[SerialRecognitionResult] = result_queue or Queue(maxsize=max(1, result_queue_size))
        self.orientation_mode = orientation_mode
        self.connection_start_time = float(connection_start_time if connection_start_time is not None else clock())
        self.poll_timeout = float(poll_timeout)
        self.clock = clock
        self.last_error: BaseException | None = None
        self.inference_ms: float | None = None
        self.average_inference_ms: float | None = None
        self.dropped_result_count = 0
        self.dropped_input_frame_count = 0
        self.processed_frames = 0
        self.last_frame: np.ndarray | None = None
        self.last_prediction: FramePrediction | None = None
        self.prediction_listener: Callable[[SerialRecognitionResult], None] | None = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._control_queue: Queue[_ControlRequest] = Queue()
        self._recognizer_lock = threading.RLock()
        self._inference_total_ms = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self.last_error = None
        self._thread = threading.Thread(target=self._run, name="RecognitionWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def reset_recognizer(self, *, wait: bool = False, timeout: float = 1.0) -> None:
        self._submit_control("reset", wait=wait, timeout=timeout)

    def calibrate(self, *, frame: np.ndarray, wait: bool = False, timeout: float = 1.0) -> None:
        self._submit_control("calibrate", frame=apply_orientation(frame, "原始"), wait=wait, timeout=timeout)

    def _submit_control(
        self,
        action: str,
        *,
        frame: np.ndarray | None = None,
        wait: bool,
        timeout: float,
    ) -> None:
        if not self.is_running:
            with self._recognizer_lock:
                self._execute_control(_ControlRequest(action=action, frame=frame))
            return
        request = _ControlRequest(action=action, frame=frame, done=threading.Event())
        self._control_queue.put(request)
        if wait:
            assert request.done is not None
            if not request.done.wait(timeout=timeout):
                raise TimeoutError(f"RecognitionWorker control request timed out: {action}")
            if request.error is not None:
                raise request.error

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._process_control_requests()
            try:
                frame = self._read_latest_frame()
            except TimeoutError:
                continue
            except BaseException as exc:
                self.last_error = exc
                break
            if frame is None:
                continue
            try:
                self._predict_frame(frame)
            except BaseException as exc:
                self.last_error = exc
                break
        self._process_control_requests()

    def _process_control_requests(self) -> None:
        while True:
            try:
                request = self._control_queue.get_nowait()
            except Empty:
                return
            try:
                with self._recognizer_lock:
                    self._execute_control(request)
            except BaseException as exc:
                request.error = exc
                self.last_error = exc
            finally:
                if request.done is not None:
                    request.done.set()

    def _execute_control(self, request: _ControlRequest) -> None:
        if request.action == "reset":
            reset = getattr(self.recognizer, "reset", None)
            if callable(reset):
                reset()
            return
        if request.action == "calibrate":
            calibrate = getattr(self.recognizer, "calibrate", None)
            if callable(calibrate):
                calibrate(frame=request.frame)
            return
        raise ValueError(f"Unknown recognition control action: {request.action}")

    def _read_latest_frame(self) -> np.ndarray | None:
        if hasattr(self.frame_source, "read_frame") and callable(self.frame_source.read_frame):
            return self.frame_source.read_frame()
        try:
            frame = self.frame_source.get(timeout=self.poll_timeout)
        except Empty as exc:
            raise TimeoutError from exc
        while True:
            try:
                frame = self.frame_source.get_nowait()
                self.dropped_input_frame_count += 1
            except Empty:
                break
        return frame

    def _predict_frame(self, frame: np.ndarray) -> None:
        oriented = apply_orientation(frame, self._current_orientation_mode())
        timestamp = max(0.0, float(self.clock() - self.connection_start_time))
        start = time.perf_counter()
        with self._recognizer_lock:
            result = self.recognizer.predict(oriented)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        record = frame_record_from_result(
            frame=oriented,
            result=result,
            frame_index=self.processed_frames,
            timestamp=timestamp,
        )
        self.processed_frames += 1
        self.inference_ms = float(elapsed_ms)
        self._inference_total_ms += self.inference_ms
        self.average_inference_ms = self._inference_total_ms / max(self.processed_frames, 1)
        self.last_frame = oriented
        self.last_prediction = record
        serial_result = SerialRecognitionResult(
            frame=oriented,
            prediction=record,
            raw_result=dict(result),
            inference_ms=self.inference_ms,
        )
        self._notify_prediction(serial_result)
        self._enqueue_result(serial_result)

    def _notify_prediction(self, result: SerialRecognitionResult) -> None:
        if self.prediction_listener is None:
            return
        try:
            self.prediction_listener(result)
        except BaseException as exc:
            self.last_error = exc

    def _enqueue_result(self, result: SerialRecognitionResult) -> None:
        try:
            self.result_queue.put_nowait(result)
            return
        except Full:
            pass
        try:
            self.result_queue.get_nowait()
            self.dropped_result_count += 1
        except Empty:
            pass
        try:
            self.result_queue.put_nowait(result)
        except Full:
            self.dropped_result_count += 1

    def _current_orientation_mode(self) -> str:
        if callable(self.orientation_mode):
            return str(self.orientation_mode())
        return str(self.orientation_mode)
