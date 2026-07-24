from __future__ import annotations

import time
import unittest
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from recognizer.data_loader import read_sensor_csv
from recognizer.frame_reader import SerialFrameReader, list_serial_ports
from recognizer.serial_protocol import PressurePacketParser
from recognizer.serial_recorder import SerialDataRecorder
from recognizer.tests.test_serial_protocol import build_packet


class FakeSerial:
    def __init__(self, chunks: list[bytes] | None = None, *, error: Exception | None = None) -> None:
        self.chunks = deque(chunks or [])
        self.error = error
        self.kwargs: dict[str, object] = {}
        self.closed = False
        self.read_calls = 0

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        if self.error is not None:
            raise self.error
        if self.chunks:
            return self.chunks.popleft()
        time.sleep(0.005)
        return b""

    def close(self) -> None:
        self.closed = True


class BlockingFakeSerial:
    def __init__(self) -> None:
        self.chunks: Queue[bytes | BaseException] = Queue()
        self.kwargs: dict[str, object] = {}
        self.closed = False
        self.read_calls = 0

    def push(self, chunk: bytes | BaseException) -> None:
        self.chunks.put(chunk)

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        try:
            chunk = self.chunks.get(timeout=0.02)
        except Empty:
            return b""
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk

    def close(self) -> None:
        self.closed = True


def wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


class SerialFrameReaderTest(unittest.TestCase):
    def make_reader(self, fake: FakeSerial, **kwargs) -> SerialFrameReader:
        def factory(**serial_kwargs):
            fake.kwargs = serial_kwargs
            return fake

        return SerialFrameReader(port="/dev/cu.TEST", timeout=0.2, serial_factory=factory, **kwargs)

    def test_reads_complete_packet_without_real_serial_hardware(self) -> None:
        fake = FakeSerial([build_packet()])
        reader = self.make_reader(fake)

        try:
            frame = reader.read_frame()
        finally:
            reader.stop()

        self.assertEqual(frame.shape, (16, 16))
        self.assertEqual(frame.dtype, np.float32)
        self.assertEqual(float(frame[1, 0]), 1.0)
        self.assertEqual(fake.kwargs["baudrate"], 460800)
        self.assertEqual(fake.kwargs["bytesize"], 8)
        self.assertEqual(fake.kwargs["parity"], "N")
        self.assertEqual(fake.kwargs["stopbits"], 1)
        self.assertFalse(fake.kwargs["xonxoff"])
        self.assertFalse(fake.kwargs["rtscts"])
        self.assertFalse(fake.kwargs["dsrdtr"])

    def test_reads_packet_split_across_serial_reads(self) -> None:
        packet = build_packet(bytes([9]) * 256)
        fake = FakeSerial([packet[:12], packet[12:141], packet[141:]])
        reader = self.make_reader(fake)

        try:
            frame = reader.read_frame()
        finally:
            reader.stop()

        self.assertEqual(float(frame[0, 0]), 9.0)
        self.assertEqual(reader.valid_frames, 1)

    def test_queue_drops_oldest_frame_when_full(self) -> None:
        packets = [
            build_packet(bytes([1]) * 256),
            build_packet(bytes([2]) * 256),
            build_packet(bytes([3]) * 256),
        ]
        fake = FakeSerial(packets)
        reader = self.make_reader(fake, queue_size=1)

        try:
            reader.start()
            wait_until(lambda: reader.valid_frames >= 3)
            frame = reader.read_frame()
        finally:
            reader.stop()

        self.assertEqual(float(frame[0, 0]), 3.0)
        self.assertEqual(reader.dropped_queue_frames, 2)

    def test_stop_is_safe_before_start_and_idempotent(self) -> None:
        fake = FakeSerial([])
        reader = self.make_reader(fake)

        reader.stop()
        reader.start()
        reader.stop()
        reader.stop()

        self.assertFalse(reader.is_running)
        self.assertTrue(fake.closed)

    def test_read_exception_is_saved_as_last_error(self) -> None:
        fake = FakeSerial(error=RuntimeError("device disconnected"))
        reader = self.make_reader(fake)

        try:
            reader.start()
            wait_until(lambda: reader.last_error is not None)
        finally:
            reader.stop()

        self.assertIn("device disconnected", str(reader.last_error))

    def test_statistics_track_received_bytes_and_parser_counts(self) -> None:
        packet = build_packet()
        fake = FakeSerial([packet])
        reader = self.make_reader(fake)

        try:
            reader.read_frame()
            stats = reader.stats()
        finally:
            reader.stop()

        self.assertGreaterEqual(stats["received_bytes"], len(packet))
        self.assertEqual(stats["valid_frames"], 1)
        self.assertEqual(stats["invalid_frames"], 0)
        self.assertEqual(stats["discarded_bytes"], 0)
        self.assertIn("current_fps", stats)

    def test_serial_reader_does_not_import_or_call_recognizer(self) -> None:
        import recognizer.frame_reader as frame_reader

        self.assertFalse(hasattr(frame_reader, "Recognizer"))

    def test_list_serial_ports_returns_all_reported_ports(self) -> None:
        ports = [
            SimpleNamespace(device="/dev/cu.usbserial-130", description="USB Serial"),
            SimpleNamespace(device="/dev/cu.Bluetooth-Incoming-Port", description="Bluetooth"),
            SimpleNamespace(device="debug-console", description="Debug Console"),
        ]

        with patch("recognizer.frame_reader._load_serial_comports", return_value=lambda: ports):
            self.assertEqual(list_serial_ports(), ports)

    def test_raw_chunk_listener_runs_before_parser_feed(self) -> None:
        packet = build_packet()
        fake = FakeSerial([packet])
        observed: list[tuple[bytes, int]] = []
        reader = self.make_reader(fake)
        reader.raw_chunk_listener = lambda chunk: observed.append((chunk, reader.parser.valid_packets))

        try:
            reader.read_frame()
        finally:
            reader.stop()

        self.assertEqual(observed, [(packet, 0)])

    def test_parsed_frame_listener_runs_before_queue_drop(self) -> None:
        packets = [
            build_packet(bytes([1]) * 256),
            build_packet(bytes([2]) * 256),
            build_packet(bytes([3]) * 256),
        ]
        fake = FakeSerial(packets)
        parsed_values: list[float] = []
        reader = self.make_reader(fake, queue_size=1)
        reader.parsed_frame_listener = lambda parsed: parsed_values.append(float(parsed.matrix[0, 0]))

        try:
            reader.start()
            wait_until(lambda: reader.valid_frames >= 3)
            frame = reader.read_frame()
        finally:
            reader.stop()

        self.assertEqual(parsed_values, [1.0, 2.0, 3.0])
        self.assertEqual(float(frame[0, 0]), 3.0)
        self.assertEqual(reader.dropped_queue_frames, 2)

    def test_begin_recording_boundary_discards_cross_boundary_packet_and_old_queue(self) -> None:
        first = build_packet(bytes([11]) * 256)
        second = build_packet(bytes([22]) * 256)
        third = build_packet(bytes([33]) * 256)
        fake = BlockingFakeSerial()
        reader = self.make_reader(fake, queue_size=8)
        raw_chunks: list[bytes] = []
        parsed_packets: list[bytes] = []

        try:
            reader.start()
            fake.push(first[:120])
            wait_until(lambda: reader.parser.buffered_bytes == 120)
            reader._enqueue_frame(np.full((16, 16), 99, dtype=np.float32))

            boundary = reader.begin_recording_boundary(
                raw_chunk_listener=raw_chunks.append,
                parsed_frame_listener=lambda parsed: parsed_packets.append(parsed.raw_packet),
            )
            fake.push(first[120:] + second + third)
            wait_until(lambda: len(parsed_packets) == 2)
        finally:
            reader.stop()

        recorded_raw = b"".join(raw_chunks)
        replayed = PressurePacketParser().feed(recorded_raw)
        self.assertEqual(boundary.buffered_bytes_cleared, 120)
        self.assertEqual(boundary.queued_frames_cleared, 1)
        self.assertEqual(recorded_raw, first[120:] + second + third)
        self.assertEqual(parsed_packets, [second, third])
        self.assertEqual([frame.raw_packet for frame in replayed], [second, third])
        self.assertEqual(reader.valid_frames, 2)

    def test_clear_pending_frames_discards_pre_rotation_queue_frames(self) -> None:
        fake = BlockingFakeSerial()
        reader = self.make_reader(fake, queue_size=8)

        try:
            reader.start()
            reader._enqueue_frame(np.full((16, 16), 1, dtype=np.float32))
            reader._enqueue_frame(np.full((16, 16), 2, dtype=np.float32))

            cleared = reader.clear_pending_frames()
        finally:
            reader.stop()

        self.assertEqual(cleared, 2)

    def test_capture_boundary_aligns_raw_stream_serial_text_and_pressure_csv(self) -> None:
        first = build_packet(bytes([41]) * 256)
        second = build_packet(bytes([42]) * 256)
        third = build_packet(bytes([43]) * 256)
        fake = BlockingFakeSerial()
        reader = self.make_reader(fake, queue_size=8)

        with TemporaryDirectory() as tmp:
            recorder = SerialDataRecorder(queue_size=64)
            try:
                reader.start()
                fake.push(first[:120])
                wait_until(lambda: reader.parser.buffered_bytes == 120)
                reader._enqueue_frame(np.full((16, 16), 99, dtype=np.float32))
                recorder.start(
                    output_root=Path(tmp),
                    label="端正坐姿",
                    trial=1,
                    serial_port="/dev/cu.TEST",
                    baudrate=460800,
                    orientation="原始",
                    model_version="v2_4_3_candidate",
                )

                reader.begin_recording_boundary(
                    raw_chunk_listener=recorder.record_raw_chunk,
                    parsed_frame_listener=recorder.record_parsed_frame,
                )
                fake.push(first[120:] + second + third)
                wait_until(lambda: recorder.stats()["valid_frames_saved"] == 2)
            finally:
                reader.stop()
                recorder.stop(serial_reader_stats_end=reader.stats())

            assert recorder.capture_dir is not None
            raw = (recorder.capture_dir / "raw_stream.bin").read_bytes()
            serial_lines = (recorder.capture_dir / "serial_raw_data.txt").read_text(encoding="ascii").splitlines()
            _timestamps, frames = read_sensor_csv(recorder.capture_dir / "pressure_frames.csv")

        replayed_packets = [frame.raw_packet for frame in PressurePacketParser().feed(raw)]
        self.assertEqual(raw, first[120:] + second + third)
        self.assertEqual(replayed_packets, [second, third])
        self.assertEqual(serial_lines, [" ".join(f"{byte:02X}" for byte in second), " ".join(f"{byte:02X}" for byte in third)])
        self.assertEqual(frames.shape, (2, 16, 16))
        self.assertEqual(float(frames[0, 0, 0]), 42.0)
        self.assertEqual(float(frames[1, 0, 0]), 43.0)


if __name__ == "__main__":
    unittest.main()
