from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lateral_subclassifier import (
    DIAGONAL_SITTING_LABEL as ORIGINAL_DIAGONAL_SITTING_LABEL,
    SIDE_LEANING_LABEL,
    STANDARD_SIDE_SITTING_LABEL,
    _prototype_label_from_diagnosis,
    lateral_physical_gate,
)
from .lateral_merged_subclassifier import (
    DIAGONAL_SITTING_LABEL,
    LATERAL_PARENT_LABELS,
    LATERAL_UNCERTAIN_LABEL,
    SIDE_SITTING_OR_LEANING_LABEL,
    merged_lateral_label,
)
from .lateral_merged_subclassifier_v241 import (
    LABEL_TAXONOMY_VERSION_V241,
    LateralMergedFineModelV241,
    TwoStageLateralMergedRecognizerV241,
    normalize_parent_lateral_display,
    resolve_final_posture_label,
)

LABEL_TAXONOMY_VERSION_V242 = LABEL_TAXONOMY_VERSION_V241
SOFT_PHYSICAL_WARNINGS = {
    "physical_gate_front_back_support_out_of_range",
}
FRONT_BACK_WARNING = "physical_gate_front_back_support_out_of_range"


class LateralMergedFineModelV242(LateralMergedFineModelV241):
    pass


class TwoStageLateralMergedRecognizerV242(TwoStageLateralMergedRecognizerV241):
    def __init__(
        self,
        parent_recognizer: object,
        lateral_model: LateralMergedFineModelV242,
        model_version: str = "v2_4_2_candidate",
        parent_model_version: str = "v2_2_candidate",
        lateral_hold_frames: int = 6,
    ) -> None:
        super().__init__(
            parent_recognizer=parent_recognizer,
            lateral_model=lateral_model,
            model_version=model_version,
            parent_model_version=parent_model_version,
            lateral_hold_frames=lateral_hold_frames,
        )

    @staticmethod
    def _physical_failures(gate_reasons: list[str]) -> list[str]:
        failures: list[str] = []
        for reason in gate_reasons:
            for token in str(reason).split(";"):
                item = token.strip()
                if item.startswith("physical_gate_"):
                    failures.append(item)
        return failures

    @staticmethod
    def _lateral_parent_candidates(parent: dict[str, Any]) -> list[str]:
        labels = []
        for key in ["label", "final_display_label", "parent_posture_label", "raw_label", "second_label"]:
            value = parent.get(key)
            if value in LATERAL_PARENT_LABELS:
                labels.append(str(value))
        return labels

    @staticmethod
    def _parent_prototype_agreement(parent: dict[str, Any]) -> bool:
        proto_label = _prototype_label_from_diagnosis(parent.get("prototype_diagnosis"))
        if proto_label not in LATERAL_PARENT_LABELS:
            return False
        proto_merged = merged_lateral_label(proto_label)
        for label in TwoStageLateralMergedRecognizerV242._lateral_parent_candidates(parent):
            if merged_lateral_label(label) == proto_merged:
                return True
        return False

    @staticmethod
    def _evidence(parent: dict[str, Any], gate_reasons: list[str]) -> dict[str, Any]:
        failures = TwoStageLateralMergedRecognizerV242._physical_failures(gate_reasons)
        soft = [item for item in failures if item in SOFT_PHYSICAL_WARNINGS]
        hard = [item for item in failures if item not in SOFT_PHYSICAL_WARNINGS]
        parent_candidates = TwoStageLateralMergedRecognizerV242._lateral_parent_candidates(parent)
        proto_label = _prototype_label_from_diagnosis(parent.get("prototype_diagnosis"))
        proto_lateral = proto_label in LATERAL_PARENT_LABELS
        agreement = TwoStageLateralMergedRecognizerV242._parent_prototype_agreement(parent)
        strong: list[str] = []
        if parent_candidates:
            strong.append("parent_lateral_candidate")
        if proto_lateral:
            strong.append("prototype_lateral_candidate")
        if agreement:
            strong.append("parent_prototype_agreement")
        if bool(parent.get("is_boundary")) and (parent_candidates or proto_lateral):
            strong.append("lateral_boundary_region")
        return {
            "parent_candidates": parent_candidates,
            "prototype_label": proto_label if proto_label in LATERAL_PARENT_LABELS else "",
            "parent_prototype_agreement": agreement,
            "strong_evidence": strong,
            "soft_warnings": soft,
            "hard_reject_reasons": hard,
            "front_back_warning": FRONT_BACK_WARNING in soft,
            "front_back_hard_reject": FRONT_BACK_WARNING in hard,
            "physical_score": len(strong) - len(hard),
        }

    @staticmethod
    def _benign_parent_lateral_gate(parent: dict[str, Any], gate_reasons: list[str]) -> tuple[bool, str]:
        evidence = TwoStageLateralMergedRecognizerV242._evidence(parent, gate_reasons)
        soft = set(evidence["soft_warnings"])
        hard = set(evidence["hard_reject_reasons"])
        if hard:
            return False, ""
        if not evidence["parent_prototype_agreement"]:
            return False, ""
        if FRONT_BACK_WARNING in soft:
            return True, "front_back_support_soft_warning; lateral_strong_parent_prototype_gate"
        return False, ""

    def _base_payload(self, parent: dict[str, Any], should_run: bool, gate_reasons: list[str], feature_map: dict[str, Any]) -> dict[str, Any]:
        payload = super()._base_payload(parent, should_run, gate_reasons, feature_map)
        evidence = self._evidence(parent, gate_reasons)
        physical_ok, physical_reasons = lateral_physical_gate(feature_map)
        soft_warning_labels = []
        if evidence["front_back_warning"]:
            soft_warning_labels.append("front_back_support")
        payload.update({
            "lateral_gate_strong_evidence": "; ".join(evidence["strong_evidence"]),
            "lateral_gate_soft_warnings": "; ".join(soft_warning_labels),
            "lateral_gate_hard_reject_reasons": "; ".join(evidence["hard_reject_reasons"]),
            "front_back_support_warning": bool(evidence["front_back_warning"]),
            "front_back_support_hard_reject": bool(evidence["front_back_hard_reject"]),
            "parent_prototype_agreement": bool(evidence["parent_prototype_agreement"]),
            "lateral_physical_evidence_score": float(evidence["physical_score"]),
            "lateral_gate_decision": "run" if should_run else "reject",
            "lateral_gate_decision_reason": "; ".join(gate_reasons) if gate_reasons else "gate_rejected_no_lateral_evidence",
            "lateral_physical_evidence_passed": bool(physical_ok),
            "lateral_physical_evidence_reasons": "; ".join(physical_reasons),
        })
        return payload


def save_lateral_merged_fine_model_v242(path: Path | str, model: LateralMergedFineModelV242) -> None:
    import joblib
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_lateral_merged_fine_model_v242(path: Path | str) -> LateralMergedFineModelV242:
    import joblib
    return joblib.load(path)


def save_lateral_merged_prototype_bank_v242(path: Path | str, model: LateralMergedFineModelV242) -> None:
    payload = {
        "labels": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
        "fallback_label": LATERAL_UNCERTAIN_LABEL,
        "label_taxonomy_version": LABEL_TAXONOMY_VERSION_V242,
        "feature_names": model.feature_names or [],
        "prototypes": {label: [proto.tolist() for proto in protos] for label, protos in model.prototypes.items()},
        "prototype_sources": model.prototype_sources,
        "prototype_subtypes": model.prototype_subtypes,
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "class_distance_centers": model.class_distance_centers or {},
        "class_distance_scales": model.class_distance_scales or {},
        "margin_thresholds": model.margin_thresholds or {},
        "distance_z_thresholds": model.distance_z_thresholds or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
