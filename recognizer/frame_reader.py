from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_loader import read_sensor_csv
from .feature_extractor import FRAME_SHAPE, as_frame


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
    baudrate: int
    rows: int = 16
    cols: int = 16
    delimiter: str = ","
    timestamp_enabled: bool = False
    frame_header: str | None = None
    frame_footer: str | None = None
    orientation: str = "normal"
    timeout: float = 1.0

    def read_frame(self) -> np.ndarray:
        raise NotImplementedError(
            "SerialFrameReader is a protocol skeleton. Provide the concrete serial frame format before enabling hardware reads."
        )


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
