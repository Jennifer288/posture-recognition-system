from __future__ import annotations

import unittest

import numpy as np

from recognizer.serial_protocol import (
    EXPECTED_LENGTH,
    FUNCTION_PRESSURE,
    HEADER,
    PACKET_SIZE,
    PAYLOAD_SIZE,
    TAIL,
    PressurePacketParser,
)


def build_packet(
    payload: bytes | None = None,
    *,
    length: int = EXPECTED_LENGTH,
    function: int = FUNCTION_PRESSURE,
    checksum: int = 0x00,
    tail: int = TAIL,
) -> bytes:
    if payload is None:
        payload = bytes(range(PAYLOAD_SIZE))
    if len(payload) != PAYLOAD_SIZE:
        raise ValueError("payload must be exactly 256 bytes")
    return (
        HEADER
        + int(length).to_bytes(2, byteorder="little")
        + bytes([function])
        + payload
        + bytes([checksum, tail])
    )


class PressurePacketParserTest(unittest.TestCase):
    def test_accepts_single_legal_packet(self) -> None:
        parser = PressurePacketParser()

        frames = parser.feed(build_packet())

        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0].raw_packet), PACKET_SIZE)
        self.assertEqual(frames[0].checksum, 0)
        self.assertEqual(parser.valid_packets, 1)
        self.assertEqual(parser.invalid_packets, 0)

    def test_joins_packet_split_across_two_feeds(self) -> None:
        parser = PressurePacketParser()
        packet = build_packet()

        self.assertEqual(parser.feed(packet[:71]), [])
        frames = parser.feed(packet[71:])

        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0].matrix, np.arange(256, dtype=np.float32).reshape((16, 16), order="F"))

    def test_joins_packet_fed_one_byte_at_a_time(self) -> None:
        parser = PressurePacketParser()
        outputs = []

        for byte in build_packet():
            outputs.extend(parser.feed(bytes([byte])))

        self.assertEqual(len(outputs), 1)
        self.assertEqual(parser.valid_packets, 1)

    def test_accepts_two_complete_packets_in_one_feed(self) -> None:
        parser = PressurePacketParser()
        first = build_packet(bytes([1]) * PAYLOAD_SIZE)
        second = build_packet(bytes([2]) * PAYLOAD_SIZE)

        frames = parser.feed(first + second)

        self.assertEqual(len(frames), 2)
        self.assertEqual(float(frames[0].matrix[0, 0]), 1.0)
        self.assertEqual(float(frames[1].matrix[0, 0]), 2.0)

    def test_keeps_half_of_next_packet_after_complete_packet(self) -> None:
        parser = PressurePacketParser()
        first = build_packet(bytes([3]) * PAYLOAD_SIZE)
        second = build_packet(bytes([4]) * PAYLOAD_SIZE)

        frames = parser.feed(first + second[:25])
        self.assertEqual(len(frames), 1)
        frames = parser.feed(second[25:])

        self.assertEqual(len(frames), 1)
        self.assertEqual(float(frames[0].matrix[0, 0]), 4.0)

    def test_discards_noise_before_header(self) -> None:
        parser = PressurePacketParser()

        frames = parser.feed(b"\x00noise\x99" + build_packet())

        self.assertEqual(len(frames), 1)
        self.assertGreater(parser.discarded_bytes, 0)

    def test_resynchronizes_after_wrong_length(self) -> None:
        parser = PressurePacketParser()
        bad = build_packet(length=EXPECTED_LENGTH + 1)

        frames = parser.feed(bad + build_packet())

        self.assertEqual(len(frames), 1)
        self.assertEqual(parser.invalid_packets, 1)

    def test_resynchronizes_after_wrong_function_code(self) -> None:
        parser = PressurePacketParser()
        bad = build_packet(function=0x02)

        frames = parser.feed(bad + build_packet())

        self.assertEqual(len(frames), 1)
        self.assertEqual(parser.invalid_packets, 1)

    def test_resynchronizes_after_wrong_tail(self) -> None:
        parser = PressurePacketParser()
        bad = build_packet(tail=0x00)

        frames = parser.feed(bad + build_packet())

        self.assertEqual(len(frames), 1)
        self.assertEqual(parser.invalid_packets, 1)

    def test_preserves_trailing_header_first_byte_for_next_feed(self) -> None:
        parser = PressurePacketParser()

        self.assertEqual(parser.feed(b"abc\x55"), [])
        frames = parser.feed(b"\xAA" + build_packet()[2:])

        self.assertEqual(len(frames), 1)

    def test_payload_may_contain_header_bytes(self) -> None:
        parser = PressurePacketParser()
        payload = bytearray([7] * PAYLOAD_SIZE)
        payload[20] = 0x55
        payload[21] = 0xAA

        frames = parser.feed(build_packet(bytes(payload)))

        self.assertEqual(len(frames), 1)
        self.assertEqual(float(frames[0].matrix[4, 1]), 0x55)
        self.assertEqual(float(frames[0].matrix[5, 1]), 0xAA)

    def test_checksum_zero_is_accepted(self) -> None:
        parser = PressurePacketParser()

        frames = parser.feed(build_packet(checksum=0x00))

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].checksum, 0x00)

    def test_payload_is_exactly_256_bytes(self) -> None:
        packet = build_packet()

        self.assertEqual(len(packet[5:261]), PAYLOAD_SIZE)
        self.assertEqual(len(packet), PACKET_SIZE)

    def test_matrix_shape_and_dtype(self) -> None:
        parser = PressurePacketParser()

        frame = parser.feed(build_packet())[0]

        self.assertEqual(frame.matrix.shape, (16, 16))
        self.assertEqual(frame.matrix.dtype, np.float32)

    def test_payload_is_restored_column_major(self) -> None:
        parser = PressurePacketParser()

        matrix = parser.feed(build_packet())[0].matrix

        self.assertEqual(float(matrix[0, 0]), 0.0)
        self.assertEqual(float(matrix[1, 0]), 1.0)
        self.assertEqual(float(matrix[15, 0]), 15.0)
        self.assertEqual(float(matrix[0, 1]), 16.0)
        self.assertEqual(float(matrix[1, 1]), 17.0)

    def test_empty_feed_returns_no_frames(self) -> None:
        parser = PressurePacketParser()

        self.assertEqual(parser.feed(b""), [])
        self.assertEqual(parser.valid_packets, 0)

    def test_clear_buffered_bytes_drops_partial_packet_without_resetting_stats(self) -> None:
        parser = PressurePacketParser()
        packet = build_packet(bytes([8]) * PAYLOAD_SIZE)

        self.assertEqual(parser.feed(b"noise" + packet[:120]), [])
        valid_before = parser.valid_packets
        invalid_before = parser.invalid_packets
        discarded_before = parser.discarded_bytes

        cleared = parser.clear_buffered_bytes()
        frames = parser.feed(packet[120:])

        self.assertEqual(cleared, 120)
        self.assertEqual(frames, [])
        self.assertEqual(parser.valid_packets, valid_before)
        self.assertEqual(parser.invalid_packets, invalid_before)
        self.assertGreaterEqual(parser.discarded_bytes, discarded_before)


if __name__ == "__main__":
    unittest.main()
