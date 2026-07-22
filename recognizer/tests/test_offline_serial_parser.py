from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from recognizer.offline_serial_parser import HEX_TEXT, RAW_BIN, parse_serial_input
from recognizer.serial_protocol import PACKET_SIZE
from recognizer.tests.test_serial_protocol import build_packet


def packet_line(packet: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in packet)


class OfflineSerialParserRawBinTest(unittest.TestCase):
    def test_raw_stream_parses_single_complete_packet(self) -> None:
        packet = build_packet()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_stream.bin"
            path.write_bytes(packet)

            result = parse_serial_input(path)

        self.assertEqual(result.selection.input_type, RAW_BIN)
        self.assertEqual(result.stats.total_bytes, PACKET_SIZE)
        self.assertEqual(result.stats.valid_packets, 1)
        self.assertEqual(result.stats.total_frames, 1)
        self.assertEqual(result.frames[0].shape, (16, 16))
        self.assertEqual(float(result.frames[0][1, 0]), 1.0)

    def test_raw_stream_handles_noise_sticky_packets_and_trailing_half_packet(self) -> None:
        first = build_packet(bytes([1]) * 256)
        second = build_packet(bytes([2]) * 256)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_stream.bin"
            path.write_bytes(b"noise" + first + second + first[:13])

            result = parse_serial_input(path, chunk_size=17)

        self.assertEqual(result.stats.valid_packets, 2)
        self.assertGreater(result.stats.discarded_bytes, 0)
        self.assertEqual(result.stats.trailing_incomplete_bytes, 13)
        self.assertIn("文件末尾存在未完成的串口半包", result.stats.parser_warnings)
        self.assertEqual(float(result.frames[0][0, 0]), 1.0)
        self.assertEqual(float(result.frames[1][0, 0]), 2.0)

    def test_capture_folder_prefers_raw_stream_and_ignores_realtime_outputs(self) -> None:
        raw_packet = build_packet(bytes([7]) * 256)
        text_packet = build_packet(bytes([9]) * 256)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "raw_stream.bin").write_bytes(raw_packet)
            (folder / "serial_raw_data.txt").write_text(packet_line(text_packet), encoding="ascii")
            (folder / "pressure_frames.csv").write_text("ignored", encoding="utf-8")
            (folder / "recognition_results.csv").write_text("ignored", encoding="utf-8")
            (folder / "metadata.json").write_text(json.dumps({"label": "人工标签"}), encoding="utf-8")

            result = parse_serial_input(folder)

        self.assertEqual(result.selection.input_path.name, "raw_stream.bin")
        self.assertEqual(result.selection.metadata["label"], "人工标签")
        self.assertEqual(float(result.frames[0][0, 0]), 7.0)
        self.assertTrue(any("pressure_frames.csv" in item for item in result.selection.warnings))
        self.assertTrue(any("recognition_results.csv" in item for item in result.selection.warnings))


class OfflineSerialParserHexTextTest(unittest.TestCase):
    def test_serial_raw_data_txt_single_line_valid_packet(self) -> None:
        packet = build_packet()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "serial_raw_data.txt"
            path.write_text(packet_line(packet) + "\n", encoding="ascii")

            result = parse_serial_input(path)

        self.assertEqual(result.selection.input_type, HEX_TEXT)
        self.assertEqual(result.stats.valid_packets, 1)
        self.assertEqual(result.raw_packets[0], packet)
        self.assertEqual(result.invalid_text_lines, [])

    def test_serial_raw_data_txt_preserves_valid_line_order_and_ignores_empty_lines(self) -> None:
        first = build_packet(bytes([3]) * 256)
        second = build_packet(bytes([4]) * 256)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "serial_raw_data.txt"
            path.write_text("\n" + packet_line(first) + "\n\n" + packet_line(second) + "\n", encoding="ascii")

            result = parse_serial_input(path)

        self.assertEqual([float(frame[0, 0]) for frame in result.frames], [3.0, 4.0])
        self.assertEqual(result.raw_packets, [first, second])

    def test_serial_raw_data_txt_records_invalid_hex_and_wrong_length(self) -> None:
        packet = build_packet()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "serial_raw_data.txt"
            path.write_text("GG\n" + "00 01\n" + packet_line(packet) + "\n", encoding="ascii")

            result = parse_serial_input(path)

        self.assertEqual(result.stats.valid_packets, 1)
        self.assertEqual(result.stats.invalid_text_line_count, 2)
        self.assertEqual([item.error_type for item in result.invalid_text_lines], ["invalid_hex", "wrong_packet_length"])

    def test_serial_raw_data_txt_classifies_invalid_header_and_tail(self) -> None:
        bad_header = bytearray(build_packet())
        bad_header[0] = 0x00
        bad_tail = bytearray(build_packet())
        bad_tail[-1] = 0x00
        good = build_packet()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "serial_raw_data.txt"
            path.write_text(
                packet_line(bytes(bad_header)) + "\n" + packet_line(bytes(bad_tail)) + "\n" + packet_line(good),
                encoding="ascii",
            )

            result = parse_serial_input(path)

        self.assertEqual(result.stats.valid_packets, 1)
        self.assertEqual([item.error_type for item in result.invalid_text_lines], ["invalid_header", "invalid_tail"])

    def test_serial_raw_data_txt_still_uses_pressure_packet_parser(self) -> None:
        bad_function = bytearray(build_packet())
        bad_function[4] = 0x02
        good = build_packet()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "serial_raw_data.txt"
            path.write_text(packet_line(bytes(bad_function)) + "\n" + packet_line(good), encoding="ascii")

            result = parse_serial_input(path)

        self.assertEqual(result.stats.valid_packets, 1)
        self.assertEqual(result.invalid_text_lines[0].error_type, "invalid_function")

    def test_serial_raw_data_txt_records_invalid_length_field(self) -> None:
        bad_length = build_packet(length=258)
        good = build_packet()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "serial_raw_data.txt"
            path.write_text(packet_line(bad_length) + "\n" + packet_line(good), encoding="ascii")

            result = parse_serial_input(path)

        self.assertEqual(result.stats.valid_packets, 1)
        self.assertEqual(result.invalid_text_lines[0].error_type, "invalid_length_field")

    def test_matrix_column_major_order_is_preserved(self) -> None:
        packet = build_packet(bytes(range(256)))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "serial_raw_data.txt"
            path.write_text(packet_line(packet), encoding="ascii")

            matrix = parse_serial_input(path).frames[0]

        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(float(matrix[0, 0]), 0.0)
        self.assertEqual(float(matrix[1, 0]), 1.0)
        self.assertEqual(float(matrix[15, 0]), 15.0)
        self.assertEqual(float(matrix[0, 1]), 16.0)
        self.assertEqual(float(matrix[1, 1]), 17.0)


if __name__ == "__main__":
    unittest.main()
