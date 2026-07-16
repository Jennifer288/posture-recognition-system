from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feature_extractor import extract_features, mirror_frame_lr
from .prototype_bank import Prototype, PrototypeBank


@dataclass(frozen=True)
class RecognizerConfig:
    boundary_margin: float = 0.0
    min_confidence: float = 0.0
    use_bank_thresholds: bool = True


@dataclass(frozen=True)
class PosturePrediction:
    label: str
    confidence: float
    second_label: str
    margin: float
    is_boundary: bool
    best_distance: float
    second_distance: float
    matched_prototype_id: str


class PrototypeRecognizer:
    def __init__(self, prototype_bank: PrototypeBank, config: RecognizerConfig | None = None) -> None:
        self.prototype_bank = prototype_bank
        self.config = config or RecognizerConfig(boundary_margin=prototype_bank.margin_threshold)

    def predict_posture(self, window: np.ndarray) -> PosturePrediction:
        feature = extract_features(window)
        mirror_feature = self._mirror_feature(window)
        distances, best_prototypes = self._class_distances(feature, mirror_feature)
        ordered = sorted(distances.items(), key=lambda item: item[1])
        label, best_distance = ordered[0]
        second_label, second_distance = ordered[1] if len(ordered) > 1 else (label, best_distance)
        margin = float(second_distance - best_distance)
        confidence = 1.0 if second_distance <= 0 else float(np.clip(1.0 - best_distance / second_distance, 0.0, 1.0))
        is_boundary = self._is_boundary(label, best_distance, margin, confidence)
        return PosturePrediction(
            label=label,
            confidence=round(confidence, 4),
            second_label=second_label,
            margin=round(margin, 4),
            is_boundary=is_boundary,
            best_distance=round(float(best_distance), 4),
            second_distance=round(float(second_distance), 4),
            matched_prototype_id=best_prototypes[label].prototype_id,
        )

    def _class_distances(
        self,
        feature: np.ndarray,
        mirror_feature: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, Prototype]]:
        feature_z = self.prototype_bank.standardized(feature)
        mirror_z = self.prototype_bank.standardized(mirror_feature)
        distances: dict[str, float] = {}
        best_prototypes: dict[str, Prototype] = {}
        for label in self.prototype_bank.labels:
            best_distance = float("inf")
            best_proto: Prototype | None = None
            for prototype in self.prototype_bank.prototypes_for_label(label):
                proto_z = self.prototype_bank.standardized(prototype.vector)
                distance = float(np.linalg.norm(feature_z - proto_z))
                if prototype.mirror_aware:
                    distance = min(distance, float(np.linalg.norm(mirror_z - proto_z)))
                if distance < best_distance:
                    best_distance = distance
                    best_proto = prototype
            if best_proto is None:
                continue
            distances[label] = best_distance
            best_prototypes[label] = best_proto
        if not distances:
            raise ValueError("No prototypes available for prediction")
        return distances, best_prototypes

    def _mirror_feature(self, window: np.ndarray) -> np.ndarray:
        arr = np.asarray(window, dtype=float)
        if arr.shape == (264,):
            return arr
        if arr.shape == (16, 16):
            return extract_features(mirror_frame_lr(arr))
        if arr.ndim == 3 and arr.shape[1:] == (16, 16):
            return extract_features(np.flip(arr, axis=2))
        return extract_features(arr)

    def _is_boundary(self, label: str, best_distance: float, margin: float, confidence: float) -> bool:
        if self.config.use_bank_thresholds:
            threshold = self.prototype_bank.class_thresholds.get(label)
            if threshold is not None and best_distance > threshold:
                return True
        if margin < self.config.boundary_margin:
            return True
        if confidence < self.config.min_confidence:
            return True
        return False


class ProbabilityRecognizer:
    def __init__(
        self,
        model: object,
        labels: list[str] | None = None,
        min_confidence: float = 0.55,
        boundary_margin: float = 0.10,
    ) -> None:
        self.model = model
        self.labels = np.asarray(labels if labels is not None else getattr(model, "classes_"))
        self.min_confidence = float(min_confidence)
        self.boundary_margin = float(boundary_margin)

    def predict_posture(self, window: np.ndarray) -> PosturePrediction:
        feature = extract_features(window).reshape(1, -1)
        proba = np.asarray(self.model.predict_proba(feature), dtype=float)[0]
        order = np.argsort(proba)[::-1]
        best_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else best_index
        confidence = float(proba[best_index])
        second_confidence = float(proba[second_index])
        margin = confidence - second_confidence
        label = str(self.labels[best_index])
        second_label = str(self.labels[second_index])
        return PosturePrediction(
            label=label,
            confidence=round(confidence, 4),
            second_label=second_label,
            margin=round(float(margin), 4),
            is_boundary=confidence < self.min_confidence or margin < self.boundary_margin,
            best_distance=round(float(1.0 - confidence), 4),
            second_distance=round(float(1.0 - second_confidence), 4),
            matched_prototype_id="probability_model",
        )
