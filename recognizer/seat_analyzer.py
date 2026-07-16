from __future__ import annotations

from collections import deque

import numpy as np

from .feature_extractor import as_frames
from .occupancy_detector import OccupancyDetector, OccupancyResult, OccupancyState
from .seat_detector import SeatDetector
from .smoothing import PredictionSmoother


class SeatAnalyzer:
    def __init__(
        self,
        recognizer: object | None = None,
        fps: float = 20.0,
        window_seconds: float = 1.5,
        settle_seconds: float = 1.0,
        occupancy_detector: OccupancyDetector | None = None,
        smoother: PredictionSmoother | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.fps = float(fps)
        self.window_frames = max(1, int(round(window_seconds * self.fps)))
        self.occupancy_detector = occupancy_detector or OccupancyDetector(fps=fps)
        self.seat_detector = SeatDetector(fps=fps, settle_seconds=settle_seconds)
        self.smoother = smoother or PredictionSmoother()
        self._recent_frames: deque[np.ndarray] = deque(maxlen=self.window_frames)

    def reset(self) -> None:
        self.seat_detector.reset()
        self.smoother.reset()
        self._recent_frames.clear()

    def update(self, frame: np.ndarray) -> dict[str, object]:
        occupancy = self.occupancy_detector.update(frame)
        if occupancy.state != OccupancyState.HUMAN:
            self.reset()
            return self._blocked_output(occupancy)

        seat = self.seat_detector.update(frame)
        self._recent_frames.append(np.asarray(frame, dtype=float))
        if not seat.stable or len(self._recent_frames) < self.window_frames:
            return self._human_waiting_output(occupancy, reason="human detected; waiting for stable window")
        return self._recognize(np.stack(list(self._recent_frames), axis=0), occupancy)

    def analyze_seat(self, window: np.ndarray) -> dict[str, object]:
        frames = as_frames(window)
        occupancy = self.occupancy_detector.analyze(frames)
        if occupancy.state != OccupancyState.HUMAN:
            return self._blocked_output(occupancy)
        if len(frames) < self.window_frames:
            return self._human_waiting_output(occupancy, reason="human detected; window is shorter than realtime stable window")
        return self._recognize(frames, occupancy)

    def _recognize(self, frames: np.ndarray, occupancy: OccupancyResult) -> dict[str, object]:
        if self.recognizer is None:
            return self._human_waiting_output(occupancy, reason="human detected but no posture recognizer is configured")
        raw = self.recognizer.predict_posture(frames)
        smoothed = self.smoother.update(raw)
        return {
            "occupancy_state": occupancy.state.value,
            "occupancy_confidence": occupancy.confidence,
            "seat_state": "HUMAN_RECOGNIZING",
            "posture": None if smoothed.get("is_boundary") else smoothed.get("label"),
            "posture_confidence": smoothed.get("confidence"),
            "second_label": smoothed.get("second_label"),
            "margin": smoothed.get("margin"),
            "is_boundary": smoothed.get("is_boundary"),
            "raw_label": smoothed.get("raw_label") or smoothed.get("label"),
            "raw_confidence": smoothed.get("raw_confidence") or smoothed.get("confidence"),
            "boundary_reason": smoothed.get("boundary_reason"),
            "reason": occupancy.reason,
            "prototype_diagnosis": smoothed.get("prototype_diagnosis"),
            "occupancy_features": occupancy_features_dict(occupancy),
            "parent_posture_label": smoothed.get("parent_posture_label"),
            "fine_posture_label": smoothed.get("fine_posture_label"),
            "final_display_label": smoothed.get("final_display_label"),
            "subclassifier_triggered": smoothed.get("subclassifier_triggered"),
            "subclassifier_gate_reason": smoothed.get("subclassifier_gate_reason"),
            "fine_confidence": smoothed.get("fine_confidence"),
            "fine_margin": smoothed.get("fine_margin"),
            "fine_boundary": smoothed.get("fine_boundary"),
            "fine_boundary_reasons": smoothed.get("fine_boundary_reasons"),
            "fine_prototype_label": smoothed.get("fine_prototype_label"),
            "fine_prototype_distance": smoothed.get("fine_prototype_distance"),
            "fallback_used": smoothed.get("fallback_used"),
            "parent_model_version": smoothed.get("parent_model_version"),
            "submodel_version": smoothed.get("submodel_version"),
            "lateral_subclassifier_triggered": smoothed.get("lateral_subclassifier_triggered"),
            "lateral_gate_reason": smoothed.get("lateral_gate_reason"),
            "lateral_posture_label": smoothed.get("lateral_posture_label"),
            "lateral_confidence": smoothed.get("lateral_confidence"),
            "lateral_margin": smoothed.get("lateral_margin"),
            "lateral_boundary": smoothed.get("lateral_boundary"),
            "lateral_boundary_reasons": smoothed.get("lateral_boundary_reasons"),
            "lateral_prototype_label": smoothed.get("lateral_prototype_label"),
            "lateral_prototype_distance": smoothed.get("lateral_prototype_distance"),
            "lateral_fallback_used": smoothed.get("lateral_fallback_used"),
            "lateral_submodel_version": smoothed.get("lateral_submodel_version"),
            "lateral_second_label": smoothed.get("lateral_second_label"),
            "lateral_second_distance": smoothed.get("lateral_second_distance"),
            "lateral_prototype_margin": smoothed.get("lateral_prototype_margin"),
            "lateral_out_of_distribution": smoothed.get("lateral_out_of_distribution"),
        }

    def _blocked_output(self, occupancy: OccupancyResult) -> dict[str, object]:
        return {
            "occupancy_state": occupancy.state.value,
            "occupancy_confidence": occupancy.confidence,
            "seat_state": occupancy.state.value,
            "posture": None,
            "posture_confidence": None,
            "second_label": None,
            "margin": None,
            "is_boundary": occupancy.state == OccupancyState.UNKNOWN,
            "reason": occupancy.reason,
            "prototype_diagnosis": None,
            "occupancy_features": occupancy_features_dict(occupancy),
        }

    def _human_waiting_output(self, occupancy: OccupancyResult, reason: str) -> dict[str, object]:
        return {
            "occupancy_state": occupancy.state.value,
            "occupancy_confidence": occupancy.confidence,
            "seat_state": "HUMAN_STABILIZING",
            "posture": None,
            "posture_confidence": None,
            "second_label": None,
            "margin": None,
            "is_boundary": False,
            "reason": reason,
            "prototype_diagnosis": None,
            "occupancy_features": occupancy_features_dict(occupancy),
        }


def analyze_seat(window: np.ndarray, recognizer: object | None = None) -> dict[str, object]:
    return SeatAnalyzer(recognizer=recognizer).analyze_seat(window)


def occupancy_features_dict(result: OccupancyResult) -> dict[str, object]:
    return {
        "total_pressure": result.total_pressure,
        "detectable_points": result.detectable_points,
        "detectable_area": result.detectable_area,
        "active_area": result.active_area,
        "connected_regions": result.connected_regions,
        "pressure_spread": result.pressure_spread,
        "active_points": result.active_points,
        "max_region_area": result.max_region_area,
        "concentration": result.concentration,
        "left_right_extent": result.left_right_extent,
        "front_back_extent": result.front_back_extent,
        "total_cv": result.total_cv,
        "cop_motion": result.cop_motion,
        "gradual_loading": result.gradual_loading,
    }
