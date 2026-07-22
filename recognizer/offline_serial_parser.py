from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .serial_protocol import (
    EXPECTED_LENGTH,
    FUNCTION_PRESSURE,
    HEADER,
    PACKET_SIZE,
    TAIL,
    ParsedPressureFrame,
    PressurePacketParser,
)


RAW_BIN = "BIN"
HEX_TEXT = "HEX_TXT"
DEFAULT_CHUNK_SIZE = 8192


@dataclass(frozen=True)
class InvalidTextLine:
    line_number: int
    error_type: str
    error_message: str
    token_count: int


@dataclass(frozen=True)
class SerialInputSelection:
    original_path: Path
    input_path: Path
    input_type: str
    metadata_path: Path | None
    metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OfflineParseStats:
    input_type: str
    total_bytes: int
    valid_packets: int
    invalid_packets: int
    discarded_bytes: int
    trailing_incomplete_bytes: int
    total_frames: int
    parser_warnings: list[str] = field(default_factory=list)
    invalid_text_line_count: int = 0


@dataclass(frozen=True)
class OfflineSerialParseResult:
    selection: SerialInputSelection
    frames: list[np.ndarray]
    raw_packets: list[bytes]
    checksums: list[int]
    stats: OfflineParseStats
    invalid_text_lines: list[InvalidTextLine] = field(default_factory=list)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_serial_input(path: str | Path) -> SerialInputSelection:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    warnings: list[str] = []
    if source.is_dir():
        raw_path = source / "raw_stream.bin"
        text_path = source / "serial_raw_data.txt"
        if raw_path.exists():
            input_path = raw_path
            input_type = RAW_BIN
        elif text_path.exists():
            input_path = text_path
            input_type = HEX_TEXT
        else:
            raise FileNotFoundError(f"{source} does not contain raw_stream.bin or serial_raw_data.txt")
        if (source / "pressure_frames.csv").exists():
            warnings.append("pressure_frames.csv存在，但离线分析不会将其作为算法输入")
        if (source / "recognition_results.csv").exists():
            warnings.append("recognition_results.csv存在，但离线分析不会将其作为算法答案")
        metadata_path = source / "metadata.json"
    else:
        input_path = source
        suffix = source.suffix.lower()
        if suffix == ".bin" or source.name == "raw_stream.bin":
            input_type = RAW_BIN
        elif suffix == ".txt" or source.name == "serial_raw_data.txt":
            input_type = HEX_TEXT
        else:
            raise ValueError(f"Unsupported serial input file type: {source.name}")
        metadata_path = source.parent / "metadata.json"
    metadata = _read_metadata(metadata_path if metadata_path.exists() else None)
    return SerialInputSelection(
        original_path=source,
        input_path=input_path,
        input_type=input_type,
        metadata_path=metadata_path if metadata_path.exists() else None,
        metadata=metadata,
        warnings=warnings,
    )


def parse_serial_input(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> OfflineSerialParseResult:
    selection = select_serial_input(path)
    if selection.input_type == RAW_BIN:
        return parse_raw_stream(selection, chunk_size=chunk_size)
    if selection.input_type == HEX_TEXT:
        return parse_serial_text(selection)
    raise ValueError(f"Unsupported input type: {selection.input_type}")


def parse_raw_stream(selection_or_path: SerialInputSelection | str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> OfflineSerialParseResult:
    selection = _coerce_selection(selection_or_path, expected_type=RAW_BIN)
    parser = PressurePacketParser()
    parsed_frames: list[ParsedPressureFrame] = []
    total_bytes = 0
    with selection.input_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            parsed_frames.extend(parser.feed(chunk))
    trailing = len(getattr(parser, "_buffer", b""))
    warnings = list(selection.warnings)
    if trailing:
        warnings.append("文件末尾存在未完成的串口半包")
    return _result_from_parsed_frames(
        selection=selection,
        parsed_frames=parsed_frames,
        total_bytes=total_bytes,
        invalid_packets=parser.invalid_packets,
        discarded_bytes=parser.discarded_bytes,
        trailing_incomplete_bytes=trailing,
        parser_warnings=warnings,
        invalid_text_lines=[],
    )


def parse_serial_text(selection_or_path: SerialInputSelection | str | Path) -> OfflineSerialParseResult:
    selection = _coerce_selection(selection_or_path, expected_type=HEX_TEXT)
    parsed_frames: list[ParsedPressureFrame] = []
    invalid_lines: list[InvalidTextLine] = []
    total_bytes = selection.input_path.stat().st_size
    discarded_bytes = 0
    parser_invalid = 0

    with selection.input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            packet, invalid = _packet_from_hex_line(line, line_number)
            if invalid is not None:
                invalid_lines.append(invalid)
                continue
            assert packet is not None
            parser = PressurePacketParser()
            frames = parser.feed(packet)
            if len(frames) != 1:
                invalid_lines.append(
                    InvalidTextLine(
                        line_number=line_number,
                        error_type="invalid_packet",
                        error_message="hex line did not produce exactly one valid pressure packet",
                        token_count=len(line.split()),
                    )
                )
            else:
                parsed_frames.extend(frames)
            parser_invalid += parser.invalid_packets
            discarded_bytes += parser.discarded_bytes

    warnings = list(selection.warnings)
    if invalid_lines:
        warnings.append(f"serial_raw_data.txt包含{len(invalid_lines)}行无效数据，已跳过")
    return _result_from_parsed_frames(
        selection=selection,
        parsed_frames=parsed_frames,
        total_bytes=total_bytes,
        invalid_packets=parser_invalid + len(invalid_lines),
        discarded_bytes=discarded_bytes,
        trailing_incomplete_bytes=0,
        parser_warnings=warnings,
        invalid_text_lines=invalid_lines,
    )


def _coerce_selection(selection_or_path: SerialInputSelection | str | Path, *, expected_type: str) -> SerialInputSelection:
    if isinstance(selection_or_path, SerialInputSelection):
        selection = selection_or_path
    else:
        selection = select_serial_input(selection_or_path)
    if selection.input_type != expected_type:
        raise ValueError(f"Expected {expected_type} input, got {selection.input_type}: {selection.input_path}")
    return selection


def _result_from_parsed_frames(
    *,
    selection: SerialInputSelection,
    parsed_frames: list[ParsedPressureFrame],
    total_bytes: int,
    invalid_packets: int,
    discarded_bytes: int,
    trailing_incomplete_bytes: int,
    parser_warnings: list[str],
    invalid_text_lines: list[InvalidTextLine],
) -> OfflineSerialParseResult:
    stats = OfflineParseStats(
        input_type=selection.input_type,
        total_bytes=int(total_bytes),
        valid_packets=len(parsed_frames),
        invalid_packets=int(invalid_packets),
        discarded_bytes=int(discarded_bytes),
        trailing_incomplete_bytes=int(trailing_incomplete_bytes),
        total_frames=len(parsed_frames),
        parser_warnings=parser_warnings,
        invalid_text_line_count=len(invalid_text_lines),
    )
    return OfflineSerialParseResult(
        selection=selection,
        frames=[np.asarray(frame.matrix, dtype=np.float32) for frame in parsed_frames],
        raw_packets=[bytes(frame.raw_packet) for frame in parsed_frames],
        checksums=[int(frame.checksum) for frame in parsed_frames],
        stats=stats,
        invalid_text_lines=invalid_text_lines,
    )


def _packet_from_hex_line(line: str, line_number: int) -> tuple[bytes | None, InvalidTextLine | None]:
    tokens = line.split()
    token_count = len(tokens)
    if any(re.fullmatch(r"[0-9A-Fa-f]{2}", token) is None for token in tokens):
        return None, InvalidTextLine(
            line_number=line_number,
            error_type="invalid_hex",
            error_message="all bytes must be two hexadecimal digits",
            token_count=token_count,
        )
    if token_count != PACKET_SIZE:
        return None, InvalidTextLine(
            line_number=line_number,
            error_type="wrong_packet_length",
            error_message=f"expected {PACKET_SIZE} bytes, got {token_count}",
            token_count=token_count,
        )
    packet = bytes(int(token, 16) for token in tokens)
    if packet[:2] != HEADER:
        return None, InvalidTextLine(line_number, "invalid_header", "packet header must be 55 AA", token_count)
    if int.from_bytes(packet[2:4], byteorder="little") != EXPECTED_LENGTH:
        return None, InvalidTextLine(line_number, "invalid_length_field", "length field must be 0x0101", token_count)
    if packet[4] != FUNCTION_PRESSURE:
        return None, InvalidTextLine(line_number, "invalid_function", "function code must be 0x01", token_count)
    if packet[-1] != TAIL:
        return None, InvalidTextLine(line_number, "invalid_tail", "packet tail must be 0x5A", token_count)
    return packet, None


def _read_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
