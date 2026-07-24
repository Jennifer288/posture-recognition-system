from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .feature_extractor import as_frame, as_frames, summarize_frame


class OccupancyState(str, Enum):
    EMPTY = "EMPTY"
    LOAD_BELOW_THRESHOLD = "LOAD_BELOW_THRESHOLD"
    OBJECT = "OBJECT"
    HUMAN = "HUMAN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class OccupancyResult:
    state: OccupancyState
    confidence: float
    reason: str
    total_pressure: float
    detectable_points: int
    detectable_area: float
    active_area: float
    connected_regions: int
    pressure_spread: float
    active_points: int
    max_region_area: int
    concentration: float
    left_right_extent: int
    front_back_extent: int
    total_cv: float
    cop_motion: float
    gradual_loading: bool


class BaselineCalibrator:
    def __init__(self, max_frames: int = 40, empty_total_threshold: float = 80.0) -> None:
        self.max_frames = max(1, int(max_frames))
        self.empty_total_threshold = float(empty_total_threshold)
        self._frames: deque[np.ndarray] = deque(maxlen=self.max_frames)
        self.baseline: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> np.ndarray:
        arr = as_frame(frame)
        if float(arr.sum()) <= self.empty_total_threshold:
            self._frames.append(arr)
            self.baseline = np.mean(np.stack(list(self._frames), axis=0), axis=0)
        return self.apply(arr)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        arr = as_frame(frame)
        if self.baseline is None:
            return arr
        return np.clip(arr - self.baseline, 0.0, None)


class OccupancyDetector:
    def __init__(
        self,
        fps: float = 20.0,
        empty_total_threshold: float = 80.0,
        occupied_total_threshold: float = 250.0,
        human_total_threshold: float = 900.0,
        detectable_value_threshold: float = 1.0,
        detectable_total_threshold: float = 10.0,
        detectable_points_min: int = 4,
        active_value_threshold: float = 15.0,
        human_active_area_min: float = 0.12,
        object_active_area_max: float = 0.075,
        history_seconds: float = 1.0,
        use_baseline: bool = True,
    ) -> None:
        self.fps = float(fps)
        self.empty_total_threshold = float(empty_total_threshold)
        self.occupied_total_threshold = float(occupied_total_threshold)
        self.human_total_threshold = float(human_total_threshold)
        self.detectable_value_threshold = float(detectable_value_threshold)
        self.detectable_total_threshold = float(detectable_total_threshold)
        self.detectable_points_min = int(detectable_points_min)
        self.active_value_threshold = float(active_value_threshold)
        self.human_active_area_min = float(human_active_area_min)
        self.object_active_area_max = float(object_active_area_max)
        self.history: deque[np.ndarray] = deque(maxlen=max(2, int(round(history_seconds * self.fps))))
        self.baseline = BaselineCalibrator(empty_total_threshold=empty_total_threshold) if use_baseline else None

    def update(self, frame: np.ndarray) -> OccupancyResult:
        arr = as_frame(frame)
        if self.baseline is not None:
            detectable_points = int((arr > self.detectable_value_threshold).sum())
            if float(arr.sum()) <= self.empty_total_threshold and detectable_points < self.detectable_points_min:
                arr = self.baseline.update(arr)
            else:
                arr = self.baseline.apply(arr)
        self.history.append(arr)
        return self.analyze(np.stack(list(self.history), axis=0))

    def analyze(self, frame_or_window: np.ndarray) -> OccupancyResult:
        frames = as_frames(frame_or_window)
        frame = frames[-1]
        temporal = frames if len(frames) > 1 else np.stack(list(self.history), axis=0) if self.history else frames
        features = self._features(frame, temporal)

        if self._is_empty(features):
            return self._result(OccupancyState.EMPTY, 0.98, "total pressure and active area are below empty thresholds", features)
        if self._is_load_below_threshold(features, temporal):
            return self._result(
                OccupancyState.LOAD_BELOW_THRESHOLD,
                0.70,
                "repeatable weak sensor response is present but below reliable occupancy threshold",
                features,
            )
        if features.total_pressure < self.occupied_total_threshold:
            return self._result(OccupancyState.UNKNOWN, 0.45, "pressure exists but is below reliable occupancy threshold", features)

        object_shape = (
            features.active_area <= self.object_active_area_max
            or features.concentration >= 0.45
            or (features.max_region_area <= 12 and features.active_area < 0.14)
        )
        human_shape = (
            features.total_pressure >= self.human_total_threshold
            and features.active_area >= self.human_active_area_min
            and features.max_region_area >= 24
            and features.left_right_extent >= 5
            and features.front_back_extent >= 5
            and features.concentration <= 0.35
        )
        reclining_human_shape = (
            features.total_pressure >= self.human_total_threshold * 1.6
            and features.active_area >= 0.065
            and features.max_region_area >= 16
            and features.left_right_extent >= 5
            and features.front_back_extent >= 5
            and features.concentration <= 0.18
        )
        compact_broad_human_support = (
            len(temporal) >= 4
            and features.total_pressure >= self.human_total_threshold
            and features.detectable_area >= self.human_active_area_min
            and features.active_area >= self.object_active_area_max * 0.5
            and features.max_region_area >= 10
            and features.left_right_extent >= 5
            and features.front_back_extent >= 5
            and features.concentration <= 0.12
        )

        if human_shape or reclining_human_shape or compact_broad_human_support:
            confidence = 0.78
            if reclining_human_shape and not human_shape:
                confidence = 0.72
            if compact_broad_human_support and not (human_shape or reclining_human_shape):
                confidence = 0.68
            if features.gradual_loading:
                confidence += 0.08
            if 0.002 <= features.total_cv <= 0.12 or features.cop_motion > 0.01:
                confidence += 0.06
            reason = (
                "high total pressure with coherent low-concentration support resembles reclining human contact"
                if reclining_human_shape and not human_shape
                else "sustained high pressure with broad low-level support resembles seated human contact"
                if compact_broad_human_support and not human_shape
                else "broad continuous pressure resembles hip/thigh support"
            )
            return self._result(OccupancyState.HUMAN, min(confidence, 0.95), reason, features)

        ambiguous_human_transition = (
            len(temporal) >= 4
            and features.total_pressure >= self.human_total_threshold
            and max(temporal.sum(axis=(1, 2))) >= self.human_total_threshold
            and features.total_cv > 0.08
            and not object_shape
        )
        if ambiguous_human_transition:
            return self._result(
                OccupancyState.HUMAN,
                0.60,
                "pressure is ambiguous but follows a recent human loading or unloading sequence",
                features,
            )

        high_pressure_history = max(temporal.sum(axis=(1, 2))) >= self.human_total_threshold
        human_transition = (
            len(temporal) >= 4
            and object_shape
            and high_pressure_history
            and features.active_area >= 0.055
            and (
                features.gradual_loading
                or features.total_cv > 0.18
                or features.total_cv > 0.08
            )
        )
        if human_transition:
            return self._result(
                OccupancyState.HUMAN,
                0.62,
                "compact pressure is changing like human sitting down or standing up, not a static object",
                features,
            )

        dynamic_compact_transition = len(temporal) >= 4 and object_shape and high_pressure_history and features.total_cv > 0.08
        if dynamic_compact_transition:
            return self._result(
                OccupancyState.UNKNOWN,
                0.55,
                "compact high-pressure contact is changing; object vs human transition evidence is unresolved",
                features,
            )

        dynamic_low_pressure_transition = len(temporal) >= 4 and object_shape and (
            features.gradual_loading or features.total_cv > 0.18
        )
        if dynamic_low_pressure_transition:
            return self._result(
                OccupancyState.UNKNOWN,
                0.52,
                "compact low-pressure contact is changing; wait for more evidence before object/human decision",
                features,
            )

        if object_shape:
            confidence = 0.72
            if features.total_cv < 0.002 and features.cop_motion < 0.01:
                confidence += 0.10
            return self._result(OccupancyState.OBJECT, min(confidence, 0.90), "pressure is compact or overly concentrated, consistent with object candidate", features)

        return self._result(OccupancyState.UNKNOWN, 0.50, "human and object evidence is insufficient or conflicting", features)

    def _features(self, frame: np.ndarray, temporal: np.ndarray) -> OccupancyResult:
        total = float(frame.sum())
        detectable_mask = frame > self.detectable_value_threshold
        detectable_points = int(detectable_mask.sum())
        detectable_area = detectable_points / 256.0
        active_mask = frame > self.active_value_threshold
        active_points = int(active_mask.sum())
        active_area = active_points / 256.0
        connected_regions, max_region_area = connected_components(active_mask)
        normalized = frame / max(total, 1.0)
        concentration = float(normalized.max()) if active_points else 0.0
        pressure_spread = float(np.sqrt((normalized * np.square(np.indices((16, 16))[0] - summarize_frame(frame).cop_y)).sum()))
        rows = np.flatnonzero(active_mask.any(axis=1))
        cols = np.flatnonzero(active_mask.any(axis=0))
        front_back_extent = int(rows[-1] - rows[0] + 1) if len(rows) else 0
        left_right_extent = int(cols[-1] - cols[0] + 1) if len(cols) else 0
        totals = temporal.sum(axis=(1, 2))
        mean_total = max(float(totals.mean()), 1.0)
        total_cv = float(totals.std() / mean_total) if len(totals) > 1 else 0.0
        summaries = [summarize_frame(item) for item in temporal]
        cop = np.asarray([(item.cop_x, item.cop_y) for item in summaries], dtype=float)
        cop_motion = float(np.sqrt(cop[:, 0].var() + cop[:, 1].var())) if len(cop) > 1 else 0.0
        recent = totals[-4:] if len(totals) >= 4 else totals
        diffs = np.diff(recent)
        positive_steps = int(np.sum(diffs > max(float(recent[-1]) * 0.03, 20.0))) if len(diffs) else 0
        gradual_loading = bool(
            len(totals) >= 4
            and recent[-1] > self.occupied_total_threshold
            and recent[0] < recent[-1] * 0.75
            and positive_steps >= 2
        )
        return OccupancyResult(
            state=OccupancyState.UNKNOWN,
            confidence=0.0,
            reason="features only",
            total_pressure=total,
            detectable_points=detectable_points,
            detectable_area=detectable_area,
            active_area=active_area,
            connected_regions=connected_regions,
            pressure_spread=pressure_spread,
            active_points=active_points,
            max_region_area=max_region_area,
            concentration=concentration,
            left_right_extent=left_right_extent,
            front_back_extent=front_back_extent,
            total_cv=total_cv,
            cop_motion=cop_motion,
            gradual_loading=gradual_loading,
        )

    def _is_empty(self, features: OccupancyResult) -> bool:
        return (
            features.total_pressure <= self.detectable_total_threshold
            or features.detectable_points < self.detectable_points_min
            or (features.total_pressure <= self.empty_total_threshold and features.active_points <= 3 and features.detectable_points < self.detectable_points_min)
        )

    def _is_load_below_threshold(self, features: OccupancyResult, temporal: np.ndarray) -> bool:
        if features.total_pressure >= self.occupied_total_threshold:
            return False
        if features.total_pressure < self.detectable_total_threshold:
            return False
        if features.detectable_points < self.detectable_points_min:
            return False
        detectable_counts = (temporal > self.detectable_value_threshold).sum(axis=(1, 2))
        repeatable_frames = int(np.sum(detectable_counts >= self.detectable_points_min))
        if repeatable_frames < min(3, len(temporal)):
            return False
        if features.active_points > 3 or features.active_area >= 0.02:
            return False
        connected, max_region = connected_components(temporal[-1] > self.detectable_value_threshold)
        return connected >= 1 and max_region >= self.detectable_points_min

    def _result(self, state: OccupancyState, confidence: float, reason: str, features: OccupancyResult) -> OccupancyResult:
        return OccupancyResult(
            state=state,
            confidence=round(float(confidence), 4),
            reason=reason,
            total_pressure=round(features.total_pressure, 4),
            detectable_points=features.detectable_points,
            detectable_area=round(features.detectable_area, 4),
            active_area=round(features.active_area, 4),
            connected_regions=features.connected_regions,
            pressure_spread=round(features.pressure_spread, 4),
            active_points=features.active_points,
            max_region_area=features.max_region_area,
            concentration=round(features.concentration, 4),
            left_right_extent=features.left_right_extent,
            front_back_extent=features.front_back_extent,
            total_cv=round(features.total_cv, 6),
            cop_motion=round(features.cop_motion, 6),
            gradual_loading=features.gradual_loading,
        )


def connected_components(mask: np.ndarray) -> tuple[int, int]:
    active = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(active, dtype=bool)
    regions = 0
    max_area = 0
    for row in range(active.shape[0]):
        for col in range(active.shape[1]):
            if not active[row, col] or visited[row, col]:
                continue
            regions += 1
            stack = [(row, col)]
            visited[row, col] = True
            area = 0
            while stack:
                r, c = stack.pop()
                area += 1
                for nr in range(max(0, r - 1), min(active.shape[0], r + 2)):
                    for nc in range(max(0, c - 1), min(active.shape[1], c + 2)):
                        if active[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            max_area = max(max_area, area)
    return regions, max_area
