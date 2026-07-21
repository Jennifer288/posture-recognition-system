from __future__ import annotations

import importlib
from queue import Queue
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from recognizer.serial_gui_core import ORIENTATION_MODES, RecognitionWorker, apply_orientation


def wait_until(predicate, *, timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def posture_payload(label: str = "端正坐姿") -> dict[str, object]:
    return {
        "occupancy": "HUMAN",
        "seat_state": "HUMAN",
        "posture": label,
        "posture_confidence": 0.91,
        "raw_label": label,
        "raw_confidence": 0.91,
        "second_label": "前倾端坐",
        "margin": 0.42,
        "is_boundary": False,
        "occupancy_features": {"total_pressure": 123.0, "active_points": 8},
    }


class FakeRecognizer:
    def __init__(self, *, fail_on_predict: bool = False, sleep_s: float = 0.0) -> None:
        self.fail_on_predict = fail_on_predict
        self.sleep_s = sleep_s
        self.predict_frames: list[np.ndarray] = []
        self.reset_count = 0
        self.calibrate_frames: list[np.ndarray] = []
        self.in_predict = False
        self.concurrent_access = False

    def predict(self, frame: np.ndarray) -> dict[str, object]:
        if self.in_predict:
            self.concurrent_access = True
        self.in_predict = True
        try:
            self.predict_frames.append(np.array(frame, copy=True))
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self.fail_on_predict:
                raise RuntimeError("predict exploded")
            return posture_payload(label=f"姿势{len(self.predict_frames)}")
        finally:
            self.in_predict = False

    def reset(self) -> None:
        if self.in_predict:
            self.concurrent_access = True
        self.reset_count += 1

    def calibrate(self, frame: np.ndarray) -> dict[str, object]:
        if self.in_predict:
            self.concurrent_access = True
        self.calibrate_frames.append(np.array(frame, copy=True))
        return {"calibrated": True}


class StepClock:
    def __init__(self, start: float = 100.0, step: float = 0.1) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class SerialGuiCoreOrientationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.arange(256, dtype=np.float32).reshape(16, 16)

    def test_apply_orientation_supports_all_eight_modes(self) -> None:
        expected = {
            "原始": self.frame,
            "上下翻转": np.flipud(self.frame),
            "左右翻转": np.fliplr(self.frame),
            "上下及左右翻转": np.flipud(np.fliplr(self.frame)),
            "转置": self.frame.T,
            "转置后上下翻转": np.flipud(self.frame.T),
            "转置后左右翻转": np.fliplr(self.frame.T),
            "转置后上下及左右翻转": np.flipud(np.fliplr(self.frame.T)),
        }

        self.assertEqual(tuple(ORIENTATION_MODES), tuple(expected))
        for mode, matrix in expected.items():
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(apply_orientation(self.frame, mode), matrix)

    def test_apply_orientation_does_not_modify_input(self) -> None:
        original = self.frame.copy()

        transformed = apply_orientation(self.frame, "左右翻转")
        transformed[0, 0] = -999.0

        np.testing.assert_array_equal(self.frame, original)

    def test_apply_orientation_returns_16x16_contiguous_array(self) -> None:
        transformed = apply_orientation(self.frame, "转置后左右翻转")

        self.assertEqual(transformed.shape, (16, 16))
        self.assertTrue(transformed.flags["C_CONTIGUOUS"])


class RecognitionWorkerTest(unittest.TestCase):
    def make_worker(
        self,
        recognizer: FakeRecognizer,
        frame_queue: Queue[np.ndarray] | None = None,
        *,
        result_queue_size: int = 3,
        clock: StepClock | None = None,
    ) -> tuple[RecognitionWorker, Queue[object], Queue[np.ndarray]]:
        source = frame_queue or Queue(maxsize=10)
        results: Queue[object] = Queue(maxsize=result_queue_size)
        worker = RecognitionWorker(
            frame_source=source,
            recognizer=recognizer,
            result_queue=results,
            orientation_mode="原始",
            connection_start_time=100.0,
            clock=clock or StepClock(),
            poll_timeout=0.02,
        )
        return worker, results, source

    def test_worker_calls_predict_in_arrival_order_when_not_backlogged(self) -> None:
        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(recognizer)

        try:
            worker.start()
            for value in (1, 2, 3):
                frames.put(np.full((16, 16), value, dtype=np.float32))
                wait_until(lambda: len(recognizer.predict_frames) >= value)
                results.get(timeout=0.5)
        finally:
            worker.stop()

        self.assertEqual([float(item[0, 0]) for item in recognizer.predict_frames], [1.0, 2.0, 3.0])

    def test_worker_timestamps_increase(self) -> None:
        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(recognizer, clock=StepClock(start=100.0, step=0.25))

        try:
            worker.start()
            frames.put(np.ones((16, 16), dtype=np.float32))
            first = results.get(timeout=0.8)
            frames.put(np.full((16, 16), 2, dtype=np.float32))
            second = results.get(timeout=0.8)
        finally:
            worker.stop()

        self.assertLess(first.prediction.timestamp, second.prediction.timestamp)

    def test_result_queue_drops_oldest_when_full(self) -> None:
        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(recognizer, result_queue_size=1)

        try:
            worker.start()
            for value in (1, 2, 3):
                frames.put(np.full((16, 16), value, dtype=np.float32))
                wait_until(lambda: len(recognizer.predict_frames) >= value)
            wait_until(lambda: worker.dropped_result_count >= 2)
        finally:
            worker.stop()

        self.assertEqual(results.qsize(), 1)
        self.assertEqual(float(results.get(timeout=0.5).frame[0, 0]), 3.0)

    def test_backlogged_frame_queue_drops_stale_unprocessed_frames(self) -> None:
        recognizer = FakeRecognizer(sleep_s=0.03)
        frames: Queue[np.ndarray] = Queue(maxsize=5)
        worker, _results, _frames = self.make_worker(recognizer, frames)

        for value in (1, 2, 3, 4, 5):
            frames.put(np.full((16, 16), value, dtype=np.float32))
        try:
            worker.start()
            wait_until(lambda: worker.dropped_input_frame_count > 0)
        finally:
            worker.stop()

        self.assertLessEqual(frames.qsize(), 5)

    def test_worker_stop_is_safe_and_idempotent(self) -> None:
        worker, _results, _frames = self.make_worker(FakeRecognizer())

        worker.start()
        worker.stop()
        worker.stop()

        self.assertFalse(worker.is_running)

    def test_predict_exception_is_saved_to_last_error(self) -> None:
        worker, _results, frames = self.make_worker(FakeRecognizer(fail_on_predict=True))

        try:
            worker.start()
            frames.put(np.ones((16, 16), dtype=np.float32))
            wait_until(lambda: worker.last_error is not None)
        finally:
            worker.stop()

        self.assertIn("predict exploded", str(worker.last_error))

    def test_reset_request_calls_recognizer_reset(self) -> None:
        recognizer = FakeRecognizer()
        worker, _results, _frames = self.make_worker(recognizer)

        try:
            worker.start()
            worker.reset_recognizer(wait=True)
        finally:
            worker.stop()

        self.assertEqual(recognizer.reset_count, 1)

    def test_calibrate_request_uses_current_frame(self) -> None:
        recognizer = FakeRecognizer()
        worker, _results, _frames = self.make_worker(recognizer)
        frame = np.arange(256, dtype=np.float32).reshape(16, 16)

        try:
            worker.start()
            worker.calibrate(frame=frame, wait=True)
        finally:
            worker.stop()

        self.assertEqual(len(recognizer.calibrate_frames), 1)
        np.testing.assert_array_equal(recognizer.calibrate_frames[0], frame)

    def test_reset_and_calibrate_do_not_run_concurrently_with_predict(self) -> None:
        recognizer = FakeRecognizer(sleep_s=0.08)
        worker, _results, frames = self.make_worker(recognizer)

        try:
            worker.start()
            frames.put(np.ones((16, 16), dtype=np.float32))
            wait_until(lambda: len(recognizer.predict_frames) == 1)
            worker.reset_recognizer(wait=True)
            worker.calibrate(frame=np.zeros((16, 16), dtype=np.float32), wait=True)
        finally:
            worker.stop()

        self.assertFalse(recognizer.concurrent_access)
        self.assertEqual(recognizer.reset_count, 1)
        self.assertEqual(len(recognizer.calibrate_frames), 1)

    def test_importing_serial_gui_does_not_open_serial_port(self) -> None:
        with patch("recognizer.frame_reader.SerialFrameReader.start", side_effect=AssertionError("opened serial")):
            module = importlib.import_module("recognizer.serial_gui")
            self.assertTrue(hasattr(module, "main"))

    def test_importing_entry_point_does_not_open_serial_port(self) -> None:
        with patch("recognizer.frame_reader.SerialFrameReader.start", side_effect=AssertionError("opened serial")):
            module = importlib.import_module("posture_serial_app_macos")
            self.assertTrue(hasattr(module, "main"))


if __name__ == "__main__":
    unittest.main()
