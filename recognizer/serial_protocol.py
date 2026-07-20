from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HEADER = b"\x55\xAA"
EXPECTED_LENGTH = 257
FUNCTION_PRESSURE = 0x01
PAYLOAD_SIZE = 256
PACKET_SIZE = 263
TAIL = 0x5A


@dataclass(frozen=True)
class ParsedPressureFrame:
    matrix: np.ndarray
    checksum: int
    raw_packet: bytes


class PressurePacketParser:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self.valid_packets = 0
        self.invalid_packets = 0
        self.discarded_bytes = 0

    def reset(self) -> None:
        self._buffer.clear()
        self.valid_packets = 0
        self.invalid_packets = 0
        self.discarded_bytes = 0

    def feed(self, chunk: bytes) -> list[ParsedPressureFrame]:
        if chunk:
            self._buffer.extend(chunk)

        frames: list[ParsedPressureFrame] = []
        while True:
            header_index = self._buffer.find(HEADER)
            if header_index < 0:
                self._discard_noise_preserving_partial_header()
                break
            if header_index > 0:
                del self._buffer[:header_index]
                self.discarded_bytes += header_index
            if len(self._buffer) < PACKET_SIZE:
                break

            packet = bytes(self._buffer[:PACKET_SIZE])
            if not self._is_candidate_packet_valid(packet):
                self.invalid_packets += 1
                del self._buffer[0]
                self.discarded_bytes += 1
                continue

            payload = packet[5:261]
            values = np.frombuffer(payload, dtype=np.uint8)
            matrix = values.reshape((16, 16), order="F").astype(np.float32)
            frames.append(ParsedPressureFrame(matrix=matrix, checksum=packet[261], raw_packet=packet))
            self.valid_packets += 1
            del self._buffer[:PACKET_SIZE]

        return frames

    def _discard_noise_preserving_partial_header(self) -> None:
        if not self._buffer:
            return
        keep_trailing_header_byte = self._buffer[-1] == HEADER[0]
        discard_count = len(self._buffer) - (1 if keep_trailing_header_byte else 0)
        if discard_count <= 0:
            return
        del self._buffer[:discard_count]
        self.discarded_bytes += discard_count

    def _is_candidate_packet_valid(self, packet: bytes) -> bool:
        if len(packet) != PACKET_SIZE:
            return False
        if packet[:2] != HEADER:
            return False
        length = int.from_bytes(packet[2:4], byteorder="little")
        if length != EXPECTED_LENGTH:
            return False
        if packet[4] != FUNCTION_PRESSURE:
            return False
        if packet[-1] != TAIL:
            return False
        return True
