from __future__ import annotations

import csv
import json
import time
import unittest
from pathlib import Path
from queue import Full
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from recognizer.csv_gui_core import FramePrediction
from recognizer.data_loader import read_sensor_csv
from recognizer.frame_orientation import apply_sensor_rotation
from recognizer.serial_gui_core import SerialRecognitionResult, apply_orientation
from recognizer.serial_protocol import ParsedPressureFrame
from recognizer.serial_recorder import SerialDataRecorder, sanitize_capture_label
from recognizer.tests.test_serial_protocol import build_packet


def make_frame(value: float = 1.0) -> np.ndarray:
    return np.full((16, 16), value, dtype=np.float32)


def make_prediction(index: int = 0, posture: str = "端正坐姿") -> SerialRecognitionResult:
    record = FramePrediction(
        timestamp=0.25 + index,
        frame_index=index,
        occupancy_state="HUMAN",
        occupancy_confidence=0.93,
        seat_state="HUMAN_RECOGNIZING",
        display_status="POSTURE",
        posture=posture,
        posture_confidence=0.91,
        raw_label=posture,
        raw_confidence=0.91,
        second_label="前倾端坐",
        margin=0.42,
        is_boundary=False,
        boundary_reason=None,
        prototype_diagnosis="端正坐姿::prototype",
        total_pressure=1234.5,
        max_pressure=88.0,
        active_points=37,
    )
    return SerialRecognitionResult(frame=make_frame(5), prediction=record, raw_result={"posture": posture}, inference_ms=7.5)


class SerialDataRecorderTest(unittest.TestCase):
    def start_recorder(self, root: Path, **kwargs) -> SerialDataRecorder:
        recorder = SerialDataRecorder(queue_size=32)
        recorder.start(
            output_root=root,
            label=kwargs.get("label", "端正坐姿"),
            trial=kwargs.get("trial", 1),
            serial_port=kwargs.get("serial_port", "COM3"),
            baudrate=kwargs.get("baudrate", 460800),
            orientation=kwargs.get("orientation", "原始"),
            sensor_rotation_degrees=kwargs.get("sensor_rotation_degrees", 0),
            model_version=kwargs.get("model_version", "v2_4_3_candidate"),
            serial_reader_stats_start={"received_bytes": 0, "valid_frames": 0},
        )
        return recorder

    def test_start_creates_unique_capture_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.start_recorder(root, label="端正坐姿", trial=1)
            first_dir = first.capture_dir
            first.stop(serial_reader_stats_end={"valid_frames": 0})

            second = self.start_recorder(root, label="端正坐姿", trial=1)
            second_dir = second.capture_dir
            second.stop(serial_reader_stats_end={"valid_frames": 0})

        self.assertIsNotNone(first_dir)
        self.assertIsNotNone(second_dir)
        self.assertNotEqual(first_dir, second_dir)
        self.assertTrue(first_dir.name.startswith("端正坐姿_1_"))
        self.assertTrue(second_dir.name.startswith("端正坐姿_1_"))

    def test_sanitizes_windows_illegal_filename_characters(self) -> None:
        self.assertEqual(sanitize_capture_label('端<正>:坐/姿\\?*"|'), "端_正__坐_姿_____")

    def test_raw_stream_preserves_exact_chunks(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp))
            recorder.record_raw_chunk(b"\x55\xaa")
            recorder.record_raw_chunk(b"noise\n\x00")
            recorder.stop()

            raw = (recorder.capture_dir / "raw_stream.bin").read_bytes()

        self.assertEqual(raw, b"\x55\xaa" + b"noise\n\x00")

    def test_start_creates_empty_serial_raw_data_text_file(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp))
            text_path = recorder.capture_dir / "serial_raw_data.txt"
            exists_while_recording = text_path.exists()
            recorder.stop()
            text = text_path.read_text(encoding="ascii")

        self.assertTrue(exists_while_recording)
        self.assertEqual(text, "")

    def test_serial_raw_data_text_is_created_with_uppercase_hex_packets(self) -> None:
        packet = build_packet(bytes(range(256)), checksum=0x00)
        parsed = ParsedPressureFrame(matrix=make_frame(1), checksum=0, raw_packet=packet)
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp))
            recorder.record_parsed_frame(parsed)
            recorder.stop()

            text_path = recorder.capture_dir / "serial_raw_data.txt"
            lines = text_path.read_text(encoding="ascii").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], " ".join(f"{byte:02X}" for byte in packet))
        self.assertNotIn(",", lines[0])
        self.assertNotIn("b'", lines[0])
        self.assertEqual(len(lines[0].split(" ")), 263)
        self.assertEqual(bytes(int(item, 16) for item in lines[0].split(" ")), packet)

    def test_serial_raw_data_text_preserves_valid_packet_order_and_ignores_noise(self) -> None:
        first = build_packet(bytes([1]) * 256)
        second = build_packet(bytes([2]) * 256)
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp))
            recorder.record_raw_chunk(b"noise" + first[:10])
            recorder.record_parsed_frame(ParsedPressureFrame(matrix=make_frame(1), checksum=0, raw_packet=first))
            recorder.record_parsed_frame(ParsedPressureFrame(matrix=make_frame(2), checksum=0, raw_packet=second))
            recorder.stop()

            lines = (recorder.capture_dir / "serial_raw_data.txt").read_text(encoding="ascii").splitlines()
            raw = (recorder.capture_dir / "raw_stream.bin").read_bytes()

        self.assertEqual(raw, b"noise" + first[:10])
        self.assertEqual(lines, [" ".join(f"{byte:02X}" for byte in first), " ".join(f"{byte:02X}" for byte in second)])

    def test_pressure_frames_csv_can_be_loaded_with_same_values(self) -> None:
        frame = np.arange(256, dtype=np.float32).reshape((16, 16))
        parsed = ParsedPressureFrame(matrix=frame, checksum=0, raw_packet=b"packet")
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp), orientation="原始")
            recorder.record_parsed_frame(parsed)
            recorder.stop()

            timestamps, frames = read_sensor_csv(recorder.capture_dir / "pressure_frames.csv")

        self.assertEqual(len(timestamps), 1)
        self.assertEqual(frames.shape, (1, 16, 16))
        np.testing.assert_array_equal(frames[0], frame)

    def test_pressure_frames_apply_locked_orientation(self) -> None:
        frame = np.arange(256, dtype=np.float32).reshape((16, 16))
        parsed = ParsedPressureFrame(matrix=frame, checksum=0, raw_packet=b"packet")
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp), orientation="左右翻转")
            recorder.record_parsed_frame(parsed)
            recorder.stop()

            _timestamps, frames = read_sensor_csv(recorder.capture_dir / "pressure_frames.csv")

        np.testing.assert_array_equal(frames[0], apply_orientation(frame, "左右翻转"))

    def test_sensor_rotation_saves_canonical_frame_and_raw_physical_frame(self) -> None:
        frame = np.arange(256, dtype=np.float32).reshape((16, 16))
        parsed = ParsedPressureFrame(matrix=frame, checksum=0, raw_packet=b"packet")
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp), orientation="原始", sensor_rotation_degrees=180)
            recorder.record_parsed_frame(parsed)
            recorder.stop()

            _canonical_timestamps, canonical_frames = read_sensor_csv(recorder.capture_dir / "pressure_frames.csv")
            _raw_timestamps, raw_frames = read_sensor_csv(recorder.capture_dir / "pressure_frames_raw.csv")

        np.testing.assert_array_equal(canonical_frames[0], apply_sensor_rotation(frame, 180))
        np.testing.assert_array_equal(raw_frames[0], frame)

    def test_sensor_rotation_runs_before_existing_orientation(self) -> None:
        frame = np.arange(256, dtype=np.float32).reshape((16, 16))
        parsed = ParsedPressureFrame(matrix=frame, checksum=0, raw_packet=b"packet")
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp), orientation="左右翻转", sensor_rotation_degrees=180)
            recorder.record_parsed_frame(parsed)
            recorder.stop()

            _timestamps, frames = read_sensor_csv(recorder.capture_dir / "pressure_frames.csv")

        expected = apply_orientation(apply_sensor_rotation(frame, 180), "左右翻转")
        np.testing.assert_array_equal(frames[0], expected)

    def test_recognition_results_csv_has_required_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp))
            recorder.record_prediction(make_prediction(index=3, posture="端正坐姿"))
            recorder.stop()

            with (recorder.capture_dir / "recognition_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame_index"], "3")
        self.assertEqual(rows[0]["posture"], "端正坐姿")
        self.assertEqual(rows[0]["inference_ms"], "7.5")
        for field in [
            "relative_time_s",
            "occupancy_state",
            "display_status",
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
        ]:
            self.assertIn(field, rows[0])

    def test_metadata_initial_and_final_fields_are_written(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp), label="前倾端坐", trial=2, orientation="上下翻转")
            initial = json.loads((recorder.capture_dir / "metadata.json").read_text(encoding="utf-8"))
            recorder.record_raw_chunk(b"abc")
            recorder.record_parsed_frame(ParsedPressureFrame(matrix=make_frame(2), checksum=0, raw_packet=b"packet"))
            recorder.record_prediction(make_prediction())
            recorder.stop(serial_reader_stats_end={"received_bytes": 3, "valid_frames": 1})
            final = json.loads((recorder.capture_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertFalse(initial["capture_completed"])
        self.assertEqual(initial["serial_text_filename"], "serial_raw_data.txt")
        self.assertEqual(initial["pressure_frames_filename"], "pressure_frames.csv")
        self.assertEqual(initial["pressure_frames_raw_filename"], "pressure_frames_raw.csv")
        self.assertEqual(initial["sensor_rotation_degrees"], 0)
        self.assertEqual(initial["orientation_transform"], "none")
        self.assertTrue(initial["physical_frame_saved"])
        self.assertTrue(initial["canonical_frame_saved"])
        self.assertEqual(initial["valid_packets_text_saved"], 0)
        self.assertEqual(final["schema_version"], "serial_capture_v1")
        self.assertEqual(final["label"], "前倾端坐")
        self.assertEqual(final["trial"], 2)
        self.assertEqual(final["orientation"], "上下翻转")
        self.assertEqual(final["raw_bytes_saved"], 3)
        self.assertEqual(final["valid_frames_saved"], 1)
        self.assertEqual(final["predictions_saved"], 1)
        self.assertEqual(final["serial_text_filename"], "serial_raw_data.txt")
        self.assertEqual(final["pressure_frames_filename"], "pressure_frames.csv")
        self.assertEqual(final["pressure_frames_raw_filename"], "pressure_frames_raw.csv")
        self.assertEqual(final["pressure_frames_format"], "canonical model-facing 16x16 pressure matrix")
        self.assertEqual(final["pressure_frames_raw_format"], "raw physical 16x16 pressure matrix from serial protocol parser")
        self.assertEqual(final["raw_stream_transform"], "none")
        self.assertEqual(final["serial_text_transform"], "none")
        self.assertEqual(final["sensor_rotation_degrees"], 0)
        self.assertEqual(final["orientation_transform"], "none")
        self.assertEqual(final["canonical_orientation"], "legacy_training_orientation")
        self.assertEqual(final["valid_packets_text_saved"], 0)
        self.assertEqual(final["serial_text_format"], "uppercase hexadecimal bytes, one valid 263-byte packet per line")
        self.assertTrue(final["capture_completed"])
        self.assertFalse(final["checksum_validation_enabled"])
        self.assertEqual(final["packet_size"], 263)
        self.assertEqual(final["matrix_shape"], [16, 16])
        self.assertEqual(final["serial_reader_stats_start"], {"received_bytes": 0, "valid_frames": 0})
        self.assertEqual(final["serial_reader_stats_end"], {"received_bytes": 3, "valid_frames": 1})

    def test_stop_drains_queue_and_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp))
            for value in range(5):
                recorder.record_raw_chunk(bytes([value]))
            recorder.stop()
            recorder.stop()

            stats = recorder.stats()
            raw = (recorder.capture_dir / "raw_stream.bin").read_bytes()

        self.assertEqual(raw, b"\x00\x01\x02\x03\x04")
        self.assertFalse(stats["is_recording"])
        self.assertEqual(stats["recorder_queue_size"], 0)

    def test_start_error_is_saved_to_last_error(self) -> None:
        with TemporaryDirectory() as tmp:
            not_a_dir = Path(tmp) / "not_a_dir"
            not_a_dir.write_text("x", encoding="utf-8")
            recorder = SerialDataRecorder()

            with self.assertRaises(Exception):
                recorder.start(
                    output_root=not_a_dir,
                    label="端正坐姿",
                    trial=1,
                    serial_port="COM3",
                    baudrate=460800,
                    orientation="原始",
                    model_version="v2_4_3_candidate",
                )

        self.assertIsNotNone(recorder.last_error)

    def test_queue_full_marks_capture_incomplete(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp))
            with patch.object(recorder._queue, "put_nowait", side_effect=Full):
                recorder.record_raw_chunk(b"lost")
            recorder.stop()
            metadata = json.loads((recorder.capture_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertGreater(metadata["recorder_dropped_events"], 0)
        self.assertFalse(metadata["data_complete"])

    def test_record_methods_are_noops_when_not_recording(self) -> None:
        recorder = SerialDataRecorder()

        recorder.record_raw_chunk(b"abc")
        recorder.record_parsed_frame(ParsedPressureFrame(matrix=make_frame(), checksum=0, raw_packet=b"packet"))
        recorder.record_prediction(make_prediction())
        recorder.stop()

        self.assertIsNone(recorder.last_error)

    def test_metadata_counts_serial_raw_data_text_lines_for_valid_packets(self) -> None:
        packet = build_packet(bytes([9]) * 256)
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp))
            recorder.record_parsed_frame(ParsedPressureFrame(matrix=make_frame(9), checksum=0, raw_packet=packet))
            recorder.stop()
            metadata = json.loads((recorder.capture_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["valid_frames_saved"], 1)
        self.assertEqual(metadata["valid_packets_text_saved"], 1)

    def test_metadata_records_sensor_rotation_180(self) -> None:
        with TemporaryDirectory() as tmp:
            recorder = self.start_recorder(Path(tmp), sensor_rotation_degrees=180)
            recorder.stop()
            metadata = json.loads((recorder.capture_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["sensor_rotation_degrees"], 180)
        self.assertEqual(metadata["orientation_transform"], "rotate_180")


if __name__ == "__main__":
    unittest.main()
