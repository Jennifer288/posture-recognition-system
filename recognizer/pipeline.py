from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .recognizer import PosturePrediction, PrototypeRecognizer
from .seat_detector import SeatDetector, SeatSnapshot


@dataclass(frozen=True)
class RealtimePrediction:
    seat: SeatSnapshot
    posture: str | None
    confidence: float
    duration_s: float
    is_boundary: bool
    prediction: PosturePrediction | None


class RealtimePosturePipeline:
    def __init__(
        self,
        recognizer: PrototypeRecognizer,
        fps: float = 20.0,
        window_seconds: float = 0.8,
        settle_seconds: float = 1.0,
    ) -> None:
        self.recognizer = recognizer
        self.fps = float(fps)
        self.window_frames = max(1, int(round(window_seconds * self.fps)))
        self.seat_detector = SeatDetector(fps=fps, settle_seconds=settle_seconds)
        self._frames: deque[np.ndarray] = deque(maxlen=self.window_frames)
        self._active_label: str | None = None
        self._active_frames = 0

    def reset(self) -> None:
        self.seat_detector.reset()
        self._frames.clear()
        self._active_label = None
        self._active_frames = 0

    def update(self, frame: np.ndarray) -> RealtimePrediction:
        seat = self.seat_detector.update(frame)
        if not seat.occupied:
            self._frames.clear()
            self._active_label = None
            self._active_frames = 0
            return RealtimePrediction(seat=seat, posture=None, confidence=0.0, duration_s=0.0, is_boundary=False, prediction=None)

        self._frames.append(np.asarray(frame, dtype=float))
        if not seat.stable or len(self._frames) < self.window_frames:
            return RealtimePrediction(seat=seat, posture=None, confidence=0.0, duration_s=0.0, is_boundary=False, prediction=None)

        prediction = self.recognizer.predict_posture(np.stack(list(self._frames), axis=0))
        if prediction.label == self._active_label:
            self._active_frames += 1
        else:
            self._active_label = prediction.label
            self._active_frames = 1
        duration = self._active_frames / self.fps
        return RealtimePrediction(
            seat=seat,
            posture=prediction.label,
            confidence=prediction.confidence,
            duration_s=duration,
            is_boundary=prediction.is_boundary,
            prediction=prediction,
        )
