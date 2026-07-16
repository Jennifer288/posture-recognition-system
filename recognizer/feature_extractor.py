from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_DIM = 264
FRAME_SHAPE = (16, 16)


@dataclass(frozen=True)
class FeatureSummary:
    total_pressure: float
    cop_x: float
    cop_y: float
    left_share: float
    right_share: float
    front_share: float
    back_share: float


def as_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=float)
    if arr.shape != FRAME_SHAPE:
        raise ValueError(f"Expected a 16x16 pressure frame, got shape {arr.shape}")
    return arr


def as_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames, dtype=float)
    if arr.shape == FRAME_SHAPE:
        return arr.reshape(1, *FRAME_SHAPE)
    if arr.ndim != 3 or arr.shape[1:] != FRAME_SHAPE:
        raise ValueError(f"Expected frames shaped (n, 16, 16), got shape {arr.shape}")
    return arr


def window_average(window: np.ndarray) -> np.ndarray:
    frames = as_frames(window)
    return frames.mean(axis=0)


def windowed_frames(frames: np.ndarray, window: int = 8, step: int = 2) -> np.ndarray:
    arr = as_frames(frames)
    if len(arr) < window:
        return arr.mean(axis=0, keepdims=True)
    return np.stack([arr[i : i + window].mean(axis=0) for i in range(0, len(arr) - window + 1, step)])


def mirror_frame_lr(frame: np.ndarray) -> np.ndarray:
    return np.flip(as_frame(frame), axis=1)


def mirror_frames_lr(frames: np.ndarray) -> np.ndarray:
    return np.flip(as_frames(frames), axis=2)


def extract_batch_features(samples: np.ndarray) -> np.ndarray:
    frames = as_frames(samples)
    totals = frames.sum(axis=(1, 2), keepdims=True)
    safe_totals = np.where(totals == 0, 1.0, totals)
    normalized = frames / safe_totals

    x = np.arange(16, dtype=float).reshape(1, 1, 16)
    y = np.arange(16, dtype=float).reshape(1, 16, 1)
    cx = (normalized * x).sum(axis=(1, 2)).reshape(-1, 1)
    cy = (normalized * y).sum(axis=(1, 2)).reshape(-1, 1)
    left = frames[:, :, :8].sum(axis=(1, 2)).reshape(-1, 1)
    right = frames[:, :, 8:].sum(axis=(1, 2)).reshape(-1, 1)
    front = frames[:, :8, :].sum(axis=(1, 2)).reshape(-1, 1)
    back = frames[:, 8:, :].sum(axis=(1, 2)).reshape(-1, 1)
    flat_safe_totals = safe_totals.reshape(-1, 1)
    lr_balance = (left - right) / flat_safe_totals
    fb_balance = (front - back) / flat_safe_totals
    flat_normalized = normalized.reshape((len(frames), -1))
    spread = flat_normalized.std(axis=1, keepdims=True)
    peak_share = flat_normalized.max(axis=1, keepdims=True)
    active_share = (frames > 20).mean(axis=(1, 2)).reshape(-1, 1)
    log_total = np.log1p(totals.reshape(-1, 1))

    features = np.hstack(
        [
            flat_normalized,
            log_total / 10.0,
            cx / 15.0,
            cy / 15.0,
            lr_balance,
            fb_balance,
            spread,
            peak_share,
            active_share,
        ]
    )
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Feature dimension changed: expected {FEATURE_DIM}, got {features.shape[1]}")
    return features


def extract_features(frame_or_window: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame_or_window, dtype=float)
    if arr.shape == (FEATURE_DIM,):
        return arr
    if arr.shape == FRAME_SHAPE:
        sample = arr.reshape(1, *FRAME_SHAPE)
    else:
        sample = window_average(arr).reshape(1, *FRAME_SHAPE)
    return extract_batch_features(sample)[0]


def summarize_frame(frame: np.ndarray) -> FeatureSummary:
    arr = as_frame(frame)
    total = float(arr.sum())
    safe_total = total if total > 0 else 1.0
    normalized = arr / safe_total
    x = np.arange(16, dtype=float).reshape(1, 16)
    y = np.arange(16, dtype=float).reshape(16, 1)
    left = float(arr[:, :8].sum() / safe_total)
    right = float(arr[:, 8:].sum() / safe_total)
    front = float(arr[:8, :].sum() / safe_total)
    back = float(arr[8:, :].sum() / safe_total)
    return FeatureSummary(
        total_pressure=total,
        cop_x=float((normalized * x).sum()),
        cop_y=float((normalized * y).sum()),
        left_share=left,
        right_share=right,
        front_share=front,
        back_share=back,
    )
