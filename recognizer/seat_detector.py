from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .feature_extractor import as_frame, summarize_frame


class SeatPhase(str, Enum):
    EMPTY = "empty"
    SITTING_DOWN = "sitting_down"
    STABILIZING = "stabilizing"
    STABLE = "stable"
    STANDING_UP = "standing_up"


@dataclass(frozen=True)
class SeatSnapshot:
    phase: SeatPhase
    total_pressure: float
    occupied: bool
    stable: bool
    occupied_duration_s: float
    stable_duration_s: float
    empty_duration_s: float
    total_cv: float
    cop_motion: float


class SeatDetector:
    def __init__(
        self,
        fps: float = 20.0,
        empty_threshold: float = 80.0,
        occupied_threshold: float = 250.0,
        settle_seconds: float = 1.0,
        stability_window_s: float = 1.0,
        total_cv_threshold: float = 0.08,
        cop_motion_threshold: float = 0.55,
    ) -> None:
        self.fps = float(fps)
        self.empty_threshold = float(empty_threshold)
        self.occupied_threshold = float(occupied_threshold)
        self.settle_frames = max(1, int(round(settle_seconds * self.fps)))
        self.history_size = max(2, int(round(stability_window_s * self.fps)))
        self.total_cv_threshold = float(total_cv_threshold)
        self.cop_motion_threshold = float(cop_motion_threshold)
        self._occupied_frames = 0
        self._stable_frames = 0
        self._empty_frames = 0
        self._phase = SeatPhase.EMPTY
        self._recent_totals: deque[float] = deque(maxlen=self.history_size)
        self._recent_cop: deque[tuple[float, float]] = deque(maxlen=self.history_size)

    def reset(self) -> None:
        self._occupied_frames = 0
        self._stable_frames = 0
        self._empty_frames = 0
        self._phase = SeatPhase.EMPTY
        self._recent_totals.clear()
        self._recent_cop.clear()

    def update(self, frame: np.ndarray) -> SeatSnapshot:
        arr = as_frame(frame)
        summary = summarize_frame(arr)
        total = summary.total_pressure
        occupied = total >= self.occupied_threshold
        nearly_empty = total <= self.empty_threshold

        if occupied:
            self._occupied_frames += 1
            self._empty_frames = 0
            self._recent_totals.append(total)
            self._recent_cop.append((summary.cop_x, summary.cop_y))
            total_cv, cop_motion = self._stability_metrics()
            stable_enough = (
                self._occupied_frames >= self.settle_frames
                and len(self._recent_totals) >= min(self.history_size, self._occupied_frames)
                and total_cv <= self.total_cv_threshold
                and cop_motion <= self.cop_motion_threshold
            )
            if stable_enough:
                self._stable_frames += 1
                self._phase = SeatPhase.STABLE
            else:
                self._stable_frames = 0
                self._phase = SeatPhase.SITTING_DOWN if self._occupied_frames < self.settle_frames else SeatPhase.STABILIZING
        elif nearly_empty:
            total_cv, cop_motion = self._stability_metrics()
            self._empty_frames += 1
            was_occupied = self._occupied_frames > 0
            self._occupied_frames = 0
            self._stable_frames = 0
            self._recent_totals.clear()
            self._recent_cop.clear()
            self._phase = SeatPhase.STANDING_UP if was_occupied else SeatPhase.EMPTY
        else:
            total_cv, cop_motion = self._stability_metrics()
            self._occupied_frames = 0
            self._stable_frames = 0
            self._phase = SeatPhase.STANDING_UP

        return SeatSnapshot(
            phase=self._phase,
            total_pressure=total,
            occupied=occupied,
            stable=self._phase == SeatPhase.STABLE,
            occupied_duration_s=self._occupied_frames / self.fps,
            stable_duration_s=self._stable_frames / self.fps,
            empty_duration_s=self._empty_frames / self.fps,
            total_cv=total_cv,
            cop_motion=cop_motion,
        )

    def _stability_metrics(self) -> tuple[float, float]:
        if len(self._recent_totals) < 2:
            return 0.0, 0.0
        totals = np.asarray(self._recent_totals, dtype=float)
        mean_total = max(float(totals.mean()), 1.0)
        total_cv = float(totals.std() / mean_total)
        cop = np.asarray(self._recent_cop, dtype=float)
        cop_motion = float(np.sqrt(cop[:, 0].var() + cop[:, 1].var()))
        return total_cv, cop_motion
