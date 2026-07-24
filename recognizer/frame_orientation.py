from __future__ import annotations

import numpy as np

from .feature_extractor import FRAME_SHAPE


SENSOR_ROTATION_0 = 0
SENSOR_ROTATION_180 = 180
SUPPORTED_SENSOR_ROTATIONS = (SENSOR_ROTATION_0, SENSOR_ROTATION_180)
SENSOR_ROTATION_OPTIONS = (
    "0°（插电口朝前 / 旧安装方式）",
    "180°（插电口朝后 / 沙发安装）",
)
SENSOR_ROTATION_LABELS = {
    SENSOR_ROTATION_0: SENSOR_ROTATION_OPTIONS[0],
    SENSOR_ROTATION_180: SENSOR_ROTATION_OPTIONS[1],
}


def apply_sensor_rotation(frame: np.ndarray, rotation_degrees: int | str) -> np.ndarray:
    """Convert a raw physical sensor frame into the legacy model-facing orientation."""
    rotation = int(rotation_degrees)
    arr = np.asarray(frame)
    if arr.shape != FRAME_SHAPE:
        raise ValueError(f"Expected a 16x16 pressure frame, got shape {arr.shape}")
    if rotation == SENSOR_ROTATION_0:
        transformed = arr
    elif rotation == SENSOR_ROTATION_180:
        transformed = arr[::-1, ::-1]
    else:
        raise ValueError(f"Unsupported sensor rotation: {rotation_degrees}")
    return np.ascontiguousarray(transformed, dtype=arr.dtype)


def orientation_transform_name(rotation_degrees: int | str) -> str:
    rotation = int(rotation_degrees)
    if rotation == SENSOR_ROTATION_0:
        return "none"
    if rotation == SENSOR_ROTATION_180:
        return "rotate_180"
    raise ValueError(f"Unsupported sensor rotation: {rotation_degrees}")


def sensor_rotation_degrees_from_label(label: str) -> int:
    text = str(label).strip()
    for degrees, option in SENSOR_ROTATION_LABELS.items():
        if text == option or text.startswith(f"{degrees}°") or text == str(degrees):
            return degrees
    raise ValueError(f"Unsupported sensor rotation label: {label}")


def sensor_rotation_label_from_degrees(degrees: int | str) -> str:
    rotation = int(degrees)
    if rotation not in SENSOR_ROTATION_LABELS:
        raise ValueError(f"Unsupported sensor rotation: {degrees}")
    return SENSOR_ROTATION_LABELS[rotation]
