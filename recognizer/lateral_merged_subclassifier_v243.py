from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .lateral_subclassifier import (
    DIAGONAL_SITTING_LABEL as ORIGINAL_DIAGONAL_SITTING_LABEL,
    SIDE_LEANING_LABEL,
    STANDARD_SIDE_SITTING_LABEL,
    extract_lateral_features,
    lateral_physical_gate,
    should_run_lateral_subclassifier,
)
from .lateral_merged_subclassifier import (
    DIAGONAL_SITTING_LABEL,
    LATERAL_UNCERTAIN_LABEL,
    SIDE_SITTING_OR_LEANING_LABEL,
)
from .lateral_merged_subclassifier_v242 import (
    FRONT_BACK_WARNING,
    LABEL_TAXONOMY_VERSION_V242,
    LateralMergedFineModelV242,
    TwoStageLateralMergedRecognizerV242,
    load_lateral_merged_fine_model_v242,
)
from .lateral_merged_subclassifier_v241 import normalize_parent_lateral_display

LABEL_TAXONOMY_VERSION_V243 = LABEL_TAXONOMY_VERSION_V242
CROSS_LEG_BACK_LABEL = "交叉腿靠背坐"


class LateralMergedFineModelV243(LateralMergedFineModelV242):
    pass


class TwoStageLateralMergedRecognizerV243(TwoStageLateralMergedRecognizerV242):
    def __init__(
        self,
        parent_recognizer: object,
        lateral_model: LateralMergedFineModelV243,
        model_version: str = "v2_4_3_candidate",
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
    def _parent_label(parent: dict[str, Any]) -> str | None:
        value = parent.get("label") or parent.get("final_display_label") or parent.get("parent_posture_label")
        return str(value) if value else None

    @staticmethod
    def _cross_leg_parent(parent: dict[str, Any]) -> bool:
        return any(parent.get(key) == CROSS_LEG_BACK_LABEL for key in ["label", "final_display_label", "parent_posture_label", "raw_label"])

    @staticmethod
    def _standard_side_signature(feature_map: dict[str, Any]) -> tuple[bool, list[str]]:
        lr_abs = abs(float(feature_map.get("left_share", 0.5)) - float(feature_map.get("right_share", 0.5)))
        row_8_11 = float(feature_map.get("row_8_11_share", 0.0))
        active_area = float(feature_map.get("active_area_ratio", 0.0))
        front = float(feature_map.get("front_share", 0.5))
        back = float(feature_map.get("back_share", 0.5))
        reasons: list[str] = []
        if lr_abs >= 0.40:
            reasons.append("strong_left_right_asymmetry")
        if row_8_11 <= 0.48:
            reasons.append("row_8_11_below_cross_leg_band")
        if active_area >= 0.22 and front >= 0.18 and back >= 0.28:
            reasons.append("stable_lateral_support_area")
        ok = {"strong_left_right_asymmetry", "row_8_11_below_cross_leg_band", "stable_lateral_support_area"}.issubset(set(reasons))
        return ok, reasons

    @staticmethod
    def _cross_leg_support(feature_map: dict[str, Any], parent: dict[str, Any]) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        if TwoStageLateralMergedRecognizerV243._cross_leg_parent(parent):
            score += 2.0
            reasons.append("parent_cross_leg")
        if parent.get("prototype_diagnosis") and CROSS_LEG_BACK_LABEL in str(parent.get("prototype_diagnosis")):
            score += 1.5
            reasons.append("parent_prototype_cross_leg")
        row_8_11 = float(feature_map.get("row_8_11_share", 0.0))
        lr_abs = abs(float(feature_map.get("left_share", 0.5)) - float(feature_map.get("right_share", 0.5)))
        if row_8_11 >= 0.49:
            score += 1.0
            reasons.append("cross_leg_back_mid_band")
        if lr_abs < 0.40:
            score += 1.0
            reasons.append("lateral_asymmetry_below_standard_side_gate")
        return score, reasons

    @staticmethod
    def _cross_leg_seat_signature(feature_map: dict[str, Any]) -> tuple[bool, list[str]]:
        """Protect true cross-leg-back pressure patterns from direct lateral gates."""
        row_4_7 = float(feature_map.get("row_4_7_share", 0.0))
        row_8_11 = float(feature_map.get("row_8_11_share", 0.0))
        row_12_15 = float(feature_map.get("row_12_15_share", 0.0))
        back = float(feature_map.get("back_share", 0.5))
        reasons: list[str] = []
        if back >= 0.55:
            reasons.append("cross_leg_back_support_high")
        if row_8_11 >= 0.47:
            reasons.append("cross_leg_mid_back_band_high")
        if row_12_15 <= 0.17:
            reasons.append("not_diagonal_rear_edge_pattern")
        if row_4_7 <= 0.26:
            reasons.append("not_standard_side_front_mid_pattern")
        required = {
            "cross_leg_back_support_high",
            "cross_leg_mid_back_band_high",
            "not_diagonal_rear_edge_pattern",
            "not_standard_side_front_mid_pattern",
        }
        return required.issubset(set(reasons)), reasons

    def _cross_leg_lateral_competition(
        self,
        parent: dict[str, Any],
        feature_vector: np.ndarray,
        feature_map: dict[str, Any],
    ) -> tuple[bool, str, dict[str, Any], dict[str, Any] | None]:
        if not self._cross_leg_parent(parent):
            return False, "not_cross_leg_parent", self._competition_payload(False, "not_cross_leg_parent", 0.0, 0.0, 0.0, [], []), None
        if bool(parent.get("subclassifier_triggered")):
            return False, "leanback_priority", self._competition_payload(False, "leanback_priority", 0.0, 0.0, 0.0, [], ["leanback_priority"]), None
        preview = self.lateral_model.predict_from_features(feature_vector)
        physical_ok, physical_reasons = lateral_physical_gate(feature_map)
        standard_side_ok, side_reasons = self._standard_side_signature(feature_map)
        cross_leg_seat_protect, cross_leg_seat_reasons = self._cross_leg_seat_signature(feature_map)
        cross_score, cross_reasons = self._cross_leg_support(feature_map, parent)
        if cross_leg_seat_protect:
            cross_score += 2.0
            cross_reasons.extend(["cross_leg_seat_signature"] + cross_leg_seat_reasons)
        lateral_score = 0.0
        lateral_reasons: list[str] = []
        if preview.get("lateral_prototype_label") == SIDE_SITTING_OR_LEANING_LABEL:
            lateral_score += 2.0
            lateral_reasons.append("nearest_lateral_prototype_side")
        if preview.get("lateral_merged_label") == SIDE_SITTING_OR_LEANING_LABEL:
            lateral_score += 2.0
            lateral_reasons.append("local_classifier_accepts_side")
        if physical_ok:
            lateral_score += 1.0
            lateral_reasons.append("lateral_physical_gate_pass")
        if standard_side_ok:
            lateral_score += 2.0
            lateral_reasons.append("standard_side_signature")
        lateral_reasons.extend(side_reasons)
        margin = lateral_score - cross_score
        allowed = bool(
            physical_ok
            and standard_side_ok
            and preview.get("lateral_prototype_label") == SIDE_SITTING_OR_LEANING_LABEL
            and preview.get("lateral_merged_label") == SIDE_SITTING_OR_LEANING_LABEL
            and not cross_leg_seat_protect
            and margin >= 1.0
        )
        reason = "conditional_cross_leg_lateral_competition; " + "; ".join(lateral_reasons) if allowed else "insufficient_lateral_competition_evidence; " + "; ".join(cross_reasons + physical_reasons + side_reasons)
        payload = self._competition_payload(allowed, reason, lateral_score, cross_score, margin, lateral_reasons, cross_reasons)
        return allowed, reason, payload, preview

    @staticmethod
    def _competition_payload(
        allowed: bool,
        reason: str,
        lateral_score: float,
        cross_score: float,
        margin: float,
        lateral_reasons: list[str],
        cross_reasons: list[str],
    ) -> dict[str, Any]:
        return {
            "cross_leg_lateral_competition_active": bool(allowed),
            "cross_leg_lateral_competition_reason": reason,
            "cross_leg_support_score": round(float(cross_score), 6),
            "lateral_support_score": round(float(lateral_score), 6),
            "lateral_vs_cross_leg_margin": round(float(margin), 6),
            "conditional_gate_override": bool(allowed),
            "conditional_gate_override_reason": reason if allowed else "",
            "cross_leg_support_reasons": "; ".join(cross_reasons),
            "lateral_support_reasons": "; ".join(lateral_reasons),
        }

    def predict_posture(self, window: np.ndarray) -> dict[str, Any]:
        parent = dict(self.parent_recognizer.predict_posture(window))
        feature_vector, feature_map = extract_lateral_features(window)
        should_run, gate_reasons = should_run_lateral_subclassifier(parent, feature_map)
        lenient_gate, lenient_reason = self._benign_parent_lateral_gate(parent, gate_reasons)
        competition_gate, competition_reason, competition_payload, preview = self._cross_leg_lateral_competition(parent, feature_vector, feature_map)
        cross_leg_seat_protect, cross_leg_seat_reasons = self._cross_leg_seat_signature(feature_map)
        if should_run and cross_leg_seat_protect and not competition_gate:
            should_run = False
            gate_reasons = list(gate_reasons) + ["cross_leg_seat_signature_protection"] + cross_leg_seat_reasons
            self._last_lateral_result = None
            self._last_lateral_features = None
            self._missed_lateral_frames = self.lateral_hold_frames + 1
        if should_run or lenient_gate or competition_gate:
            self._candidate_lateral_frames += 1
        else:
            self._candidate_lateral_frames = 0
        if not should_run and lenient_gate and self._candidate_lateral_frames >= 3:
            should_run = True
            gate_reasons = list(gate_reasons) + [lenient_reason]
        if not should_run and competition_gate and self._candidate_lateral_frames >= 3:
            should_run = True
            gate_reasons = list(gate_reasons) + [competition_reason]
        payload = self._base_payload(parent, should_run, gate_reasons, feature_map)
        payload.update(competition_payload)
        payload["cross_leg_seat_protection_active"] = bool(cross_leg_seat_protect and not competition_gate)
        payload["cross_leg_seat_protection_reasons"] = "; ".join(cross_leg_seat_reasons)
        payload["model_version"] = self.model_version
        payload["lateral_submodel_version"] = self.lateral_submodel_version
        payload["final_display_label"] = payload.get("final_display_label") or payload.get("label")
        if bool(parent.get("subclassifier_triggered")):
            payload["lateral_gate_reason"] = "leanback_priority"
            payload["final_selected_branch"] = "leanback_priority"
            return payload
        if should_run:
            lateral = preview if preview is not None and competition_gate else self.lateral_model.predict_from_features(feature_vector)
            lateral["lateral_feature_summary"] = feature_map
            self._last_lateral_result = dict(lateral)
            self._last_lateral_features = dict(feature_map)
            self._missed_lateral_frames = 0
            payload.update(lateral)
            resolved = self._apply_lateral_resolution(payload, parent, lateral, temporal_state="active")
            resolved.update(competition_payload)
            resolved["final_selected_branch"] = resolved.get("selected_final_branch") or resolved.get("selected_branch")
            return resolved
        if self._last_lateral_result is not None:
            self._missed_lateral_frames += 1
            if self._missed_lateral_frames <= self.lateral_hold_frames:
                lateral = dict(self._last_lateral_result)
                reasons = list(lateral.get("lateral_boundary_reasons") or [])
                if lateral.get("lateral_boundary") and "temporal_lateral_hold" not in reasons:
                    reasons.append("temporal_lateral_hold")
                lateral["lateral_boundary_reasons"] = reasons
                payload.update(lateral)
                payload["lateral_subclassifier_triggered"] = True
                payload["lateral_gate_reason"] = "lateral_temporal_hold"
                resolved = self._apply_lateral_resolution(payload, parent, lateral, temporal_state="hold")
                resolved.update(competition_payload)
                resolved["final_selected_branch"] = resolved.get("selected_final_branch") or resolved.get("selected_branch")
                return resolved
        if self._missed_lateral_frames > self.lateral_hold_frames:
            self._last_lateral_result = None
            self._last_lateral_features = None
            self._missed_lateral_frames = 0
        normalized = normalize_parent_lateral_display(payload, lateral_gate_active=False, physical_features=feature_map, temporal_state="inactive")
        normalized.update(competition_payload)
        normalized["final_display_label"] = normalized.get("final_display_label") or normalized.get("label")
        normalized["final_selected_branch"] = normalized.get("selected_final_branch") or normalized.get("selected_branch")
        return normalized

    def _base_payload(self, parent: dict[str, Any], should_run: bool, gate_reasons: list[str], feature_map: dict[str, Any]) -> dict[str, Any]:
        payload = super()._base_payload(parent, should_run, gate_reasons, feature_map)
        payload["model_version"] = self.model_version
        payload["lateral_submodel_version"] = self.lateral_submodel_version
        payload.setdefault("cross_leg_lateral_competition_active", False)
        payload.setdefault("cross_leg_lateral_competition_reason", "not_evaluated")
        payload.setdefault("cross_leg_support_score", 0.0)
        payload.setdefault("lateral_support_score", 0.0)
        payload.setdefault("lateral_vs_cross_leg_margin", 0.0)
        payload.setdefault("conditional_gate_override", False)
        payload.setdefault("conditional_gate_override_reason", "")
        payload.setdefault("final_selected_branch", payload.get("selected_final_branch") or payload.get("selected_branch"))
        return payload


def save_lateral_merged_fine_model_v243(path: Path | str, model: LateralMergedFineModelV243) -> None:
    import joblib
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_lateral_merged_fine_model_v243(path: Path | str) -> LateralMergedFineModelV243:
    model = load_lateral_merged_fine_model_v242(path)
    if isinstance(model, LateralMergedFineModelV243):
        return model
    return LateralMergedFineModelV243(
        prototypes=model.prototypes,
        prototype_sources=model.prototype_sources,
        prototype_subtypes=model.prototype_subtypes,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        class_distance_centers=model.class_distance_centers,
        class_distance_scales=model.class_distance_scales,
        margin_thresholds=model.margin_thresholds,
        distance_z_thresholds=model.distance_z_thresholds,
        confidence_threshold=model.confidence_threshold,
        classifier=model.classifier,
        feature_names=model.feature_names,
        submodel_version="lateral_merged_subclassifier_v2_4_3_candidate",
    )


def save_lateral_merged_prototype_bank_v243(path: Path | str, model: LateralMergedFineModelV243) -> None:
    payload = {
        "labels": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
        "fallback_label": LATERAL_UNCERTAIN_LABEL,
        "label_taxonomy_version": LABEL_TAXONOMY_VERSION_V243,
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
        "cross_leg_lateral_competition": {
            "enabled": True,
            "parent_label": CROSS_LEG_BACK_LABEL,
            "requires": [
                "stable_human",
                "no_leanback_priority",
                "local_lateral_prototype_side",
                "local_classifier_accepts_side",
                "lateral_physical_gate_pass",
                "standard_side_signature",
                "lateral_support_score_minus_cross_leg_support_score >= 1.0",
            ],
            "hard_protection": [
                "insufficient_lateral_competition_evidence",
                "physical_gate_failure",
                "standard_side_signature_missing",
            ],
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
