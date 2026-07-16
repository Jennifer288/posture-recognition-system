from __future__ import annotations

from pathlib import Path

import numpy as np

from .model_artifact import load_model_bundle
from .prototype_bank import PrototypeBank
from .recognizer import ProbabilityRecognizer, PrototypeRecognizer


class HybridPostureRecognizer:
    def __init__(
        self,
        rf_recognizer: ProbabilityRecognizer,
        prototype_recognizer: PrototypeRecognizer | None = None,
        prototype_conflict_margin: float = 0.18,
        prototype_boundary_confidence_gate: float = 0.65,
    ) -> None:
        self.rf_recognizer = rf_recognizer
        self.prototype_recognizer = prototype_recognizer
        self.prototype_conflict_margin = float(prototype_conflict_margin)
        self.prototype_boundary_confidence_gate = float(prototype_boundary_confidence_gate)

    def predict_posture(self, window: np.ndarray) -> dict[str, object]:
        rf = self.rf_recognizer.predict_posture(window)
        diagnosis = None
        prototype_boundary = False
        prototype_conflict = False
        boundary_reasons: list[str] = []
        if rf.confidence < self.rf_recognizer.min_confidence:
            boundary_reasons.append(f"RF confidence<{self.rf_recognizer.min_confidence:.2f}")
        if rf.margin < self.rf_recognizer.boundary_margin:
            boundary_reasons.append(f"RF margin<{self.rf_recognizer.boundary_margin:.2f}")
        if self.prototype_recognizer is not None:
            proto = self.prototype_recognizer.predict_posture(window)
            prototype_conflict = proto.label != rf.label and rf.margin <= self.prototype_conflict_margin
            prototype_boundary = proto.is_boundary and (
                proto.label != rf.label or rf.confidence < self.prototype_boundary_confidence_gate
            )
            if prototype_boundary:
                boundary_reasons.append("Prototype boundary")
            if prototype_conflict:
                boundary_reasons.append(f"Prototype/RF conflict with RF margin<={self.prototype_conflict_margin:.2f}")
            diagnosis = {
                "label": proto.label,
                "confidence": proto.confidence,
                "second_label": proto.second_label,
                "margin": proto.margin,
                "is_boundary": proto.is_boundary,
                "matched_prototype_id": proto.matched_prototype_id,
                "agrees_with_rf": proto.label == rf.label,
            }
        is_boundary = bool(rf.is_boundary or prototype_boundary or prototype_conflict)
        return {
            "label": rf.label,
            "confidence": rf.confidence,
            "second_label": rf.second_label,
            "margin": rf.margin,
            "is_boundary": is_boundary,
            "boundary_reasons": boundary_reasons,
            "prototype_diagnosis": diagnosis,
        }


def load_hybrid_recognizer(
    rf_model_path: Path | str,
    prototype_bank_path: Path | str | None = None,
    min_confidence: float = 0.55,
    boundary_margin: float = 0.10,
) -> HybridPostureRecognizer:
    bundle = load_model_bundle(rf_model_path)
    rf = ProbabilityRecognizer(bundle["model"], min_confidence=min_confidence, boundary_margin=boundary_margin)
    prototype = None
    if prototype_bank_path is not None:
        prototype = PrototypeRecognizer(PrototypeBank.load(prototype_bank_path))
    return HybridPostureRecognizer(rf_recognizer=rf, prototype_recognizer=prototype)
