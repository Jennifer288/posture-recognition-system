from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .lateral_subclassifier import (
    DIAGONAL_SITTING_LABEL as ORIGINAL_DIAGONAL_SITTING_LABEL,
    LATERAL_BOUNDARY_LABEL as ORIGINAL_LATERAL_BOUNDARY_LABEL,
    LATERAL_DISPLAY_CONFIDENCE,
    LATERAL_DISPLAY_MARGIN,
    LATERAL_FEATURE_NAMES,
    SIDE_LEANING_LABEL,
    STANDARD_SIDE_SITTING_LABEL,
    extract_lateral_features,
    lateral_physical_gate,
    should_run_lateral_subclassifier,
)
from .lateral_merged_subclassifier import (
    DIAGONAL_SITTING_LABEL,
    LABEL_TAXONOMY_VERSION,
    LATERAL_PARENT_LABELS,
    LATERAL_UNCERTAIN_LABEL,
    SIDE_SITTING_OR_LEANING_LABEL,
    SIDE_SOURCE_LABELS,
    merged_lateral_label,
)

LABEL_TAXONOMY_VERSION_V241 = "lateral_merge_v1"


def should_normalize_parent_lateral_label(
    parent_result: dict[str, Any],
    lateral_gate_active: bool,
    physical_features: dict[str, Any] | None,
    temporal_state: str,
) -> tuple[bool, dict[str, Any]]:
    parent_label = parent_result.get("label") or parent_result.get("final_display_label")
    physical_ok, physical_reasons = lateral_physical_gate(physical_features or {}) if physical_features else (False, ["physical_features_missing"])
    candidate_evidence = any(
        parent_result.get(key) in {STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL, DIAGONAL_SITTING_LABEL}
        for key in ["raw_label", "second_label", "parent_posture_label", "label"]
    )
    temporal_ok = temporal_state in {"active", "hold", "stable"}
    allowed = bool(parent_label in SIDE_SOURCE_LABELS and lateral_gate_active and physical_ok and temporal_ok and candidate_evidence)
    reason = "stable_lateral_gate_and_physical_evidence" if allowed else "insufficient_stable_lateral_evidence"
    return allowed, {
        "lateral_normalization_applied": allowed,
        "lateral_normalization_reason": reason,
        "lateral_normalization_confidence": 1.0 if allowed else 0.0,
        "lateral_physical_evidence_passed": bool(physical_ok),
        "lateral_physical_evidence_reasons": "; ".join(physical_reasons),
        "selected_final_branch": "parent_lateral_label_normalized" if allowed else "parent",
        "final_override_reason": reason,
    }


def normalize_parent_lateral_display(
    parent_result: dict[str, Any],
    lateral_gate_active: bool = False,
    physical_features: dict[str, Any] | None = None,
    temporal_state: str = "inactive",
) -> dict[str, Any]:
    payload = dict(parent_result)
    parent_label = payload.get("label") or payload.get("final_display_label")
    payload["label_taxonomy_version"] = LABEL_TAXONOMY_VERSION_V241
    if parent_label in {STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL, DIAGONAL_SITTING_LABEL}:
        payload["parent_raw_lateral_label"] = parent_label
    allowed, evidence = should_normalize_parent_lateral_label(payload, lateral_gate_active, physical_features, temporal_state)
    payload.update(evidence)
    if allowed:
        payload["label"] = SIDE_SITTING_OR_LEANING_LABEL
        payload["final_display_label"] = SIDE_SITTING_OR_LEANING_LABEL
        payload["selected_branch"] = "parent_lateral_label_normalized"
        payload["final_priority_branch"] = "parent_lateral_label_normalized"
        payload["override_reason"] = "v2_4_1_label_merge_normalized_parent_with_evidence"
    else:
        payload.setdefault("selected_branch", "parent")
        payload.setdefault("final_priority_branch", "parent")
        payload.setdefault("override_reason", "no_lateral_normalization_without_evidence")
    return payload


@dataclass
class LateralMergedFineModelV241:
    prototypes: dict[str, list[np.ndarray]]
    prototype_sources: dict[str, list[str]]
    prototype_subtypes: dict[str, list[str]]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    class_distance_centers: dict[str, float] | None = None
    class_distance_scales: dict[str, float] | None = None
    margin_thresholds: dict[str, float] | None = None
    distance_z_thresholds: dict[str, float] | None = None
    confidence_threshold: float = 0.45
    classifier: object | None = None
    feature_names: list[str] | None = None
    submodel_version: str = "lateral_merged_subclassifier_v2_4_1_candidate"

    def predict(self, window: np.ndarray) -> dict[str, Any]:
        features, feature_map = extract_lateral_features(window)
        result = self.predict_from_features(features)
        result["lateral_feature_summary"] = feature_map
        return result

    def predict_from_features(self, features: np.ndarray) -> dict[str, Any]:
        vector = np.asarray(features, dtype=float).reshape(-1)
        scaled = self._scale(vector)
        ordered = self._prototype_distances(scaled)
        if not ordered:
            raise ValueError("LateralMergedFineModelV241 requires at least one prototype")
        best_label, best_source, best_subtype, best_distance = ordered[0]
        second_label, second_source, second_subtype, second_distance = self._second_other_label(ordered, best_label)
        prototype_margin = float(second_distance - best_distance)
        classifier_label = None
        classifier_confidence = None
        classifier_margin = None
        if self.classifier is not None:
            classifier_label, classifier_confidence, classifier_margin = self._classifier_prediction(scaled.reshape(1, -1))
        resolved_label = str(classifier_label or best_label)
        reasons: list[str] = []
        if classifier_label is not None and classifier_label != best_label:
            resolved_label = LATERAL_UNCERTAIN_LABEL
            reasons.append("classifier_prototype_conflict")
        distance_z = self._distance_z(best_label, best_distance)
        z_threshold = float((self.distance_z_thresholds or {}).get(best_label, 3.0))
        margin_threshold = float((self.margin_thresholds or {}).get(best_label, 0.15))
        if distance_z > z_threshold:
            reasons.append("out_of_distribution")
        if prototype_margin < margin_threshold:
            reasons.append("low_prototype_margin")
        if classifier_margin is not None and classifier_margin < margin_threshold:
            reasons.append("low_classifier_margin")
        confidence = classifier_confidence if classifier_confidence is not None else float(1.0 / (1.0 + max(distance_z, 0.0)))
        if classifier_confidence is not None and classifier_confidence < self.confidence_threshold:
            reasons.append("low_classifier_confidence")
        if classifier_confidence is None and confidence < self.confidence_threshold:
            reasons.append("low_prototype_confidence")
        if {best_label, second_label} == {SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL} and prototype_margin < margin_threshold:
            reasons.append("side_vs_diagonal_overlap")
        if resolved_label not in {SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL, LATERAL_UNCERTAIN_LABEL}:
            resolved_label = LATERAL_UNCERTAIN_LABEL
            reasons.append("unknown_lateral_label")
        lateral_boundary = bool(reasons) or resolved_label == LATERAL_UNCERTAIN_LABEL
        final_label = LATERAL_UNCERTAIN_LABEL if lateral_boundary else resolved_label
        margin = classifier_margin if classifier_margin is not None else prototype_margin
        return {
            "lateral_merged_label": final_label,
            "lateral_posture_label": final_label,
            "final_display_label": final_label,
            "lateral_confidence": round(float(confidence), 6),
            "lateral_margin": round(float(margin), 6),
            "lateral_boundary": lateral_boundary,
            "lateral_boundary_reasons": sorted(set(reasons)),
            "lateral_prototype_label": best_label,
            "lateral_prototype_subtype": best_subtype,
            "lateral_prototype_source": best_source,
            "lateral_prototype_distance": round(float(best_distance), 6),
            "lateral_distance_z": round(float(distance_z), 6),
            "lateral_second_label": second_label,
            "lateral_second_subtype": second_subtype,
            "lateral_second_prototype_source": second_source,
            "lateral_second_distance": round(float(second_distance), 6),
            "lateral_prototype_margin": round(float(prototype_margin), 6),
            "lateral_fallback_used": final_label == LATERAL_UNCERTAIN_LABEL,
            "lateral_out_of_distribution": "out_of_distribution" in reasons,
            "lateral_classifier_label": classifier_label,
            "label_taxonomy_version": LABEL_TAXONOMY_VERSION_V241,
        }

    def _prototype_distances(self, scaled: np.ndarray) -> list[tuple[str, str, str, float]]:
        # Aggregate by class using each class's nearest prototype. This prevents
        # the merged side class from gaining weight merely because it has more
        # prototypes than the diagonal class.
        by_label: dict[str, tuple[str, str, float]] = {}
        for label, protos in self.prototypes.items():
            sources = self.prototype_sources.get(label, [])
            subtypes = self.prototype_subtypes.get(label, [])
            best: tuple[str, str, float] | None = None
            for idx, proto in enumerate(protos):
                source = sources[idx] if idx < len(sources) else f"{label}::prototype_{idx}"
                subtype = subtypes[idx] if idx < len(subtypes) else label
                distance = float(np.linalg.norm(scaled - self._scale(proto)))
                if best is None or distance < best[2]:
                    best = (source, subtype, distance)
            if best is not None:
                by_label[label] = best
        rows = [(label, source, subtype, distance) for label, (source, subtype, distance) in by_label.items()]
        return sorted(rows, key=lambda item: item[3])

    def _scale(self, vector: np.ndarray) -> np.ndarray:
        scale = np.where(np.abs(self.feature_scale) < 1e-9, 1.0, self.feature_scale)
        return (np.asarray(vector, dtype=float) - self.feature_mean) / scale

    def _distance_z(self, label: str, distance: float) -> float:
        center = float((self.class_distance_centers or {}).get(label, 0.0))
        scale = max(float((self.class_distance_scales or {}).get(label, 1.0)), 1e-6)
        return max(0.0, (float(distance) - center) / scale)

    @staticmethod
    def _second_other_label(ordered: list[tuple[str, str, str, float]], best_label: str) -> tuple[str, str, str, float]:
        for label, source, subtype, distance in ordered[1:]:
            if label != best_label:
                return label, source, subtype, distance
        label, source, subtype, distance = ordered[0]
        return label, source, subtype, distance + 1.0

    def _classifier_prediction(self, features: np.ndarray) -> tuple[str, float, float]:
        if hasattr(self.classifier, "predict_proba"):
            probabilities = np.asarray(self.classifier.predict_proba(features), dtype=float)[0]
            classes = [str(item) for item in self.classifier.classes_]
            order = np.argsort(probabilities)[::-1]
            top = int(order[0])
            second = int(order[1]) if len(order) > 1 else top
            return classes[top], float(probabilities[top]), float(probabilities[top] - probabilities[second])
        if hasattr(self.classifier, "decision_function"):
            scores = np.asarray(self.classifier.decision_function(features), dtype=float)
            classes = [str(item) for item in self.classifier.classes_]
            if scores.ndim == 1:
                score = float(scores[0])
                label = classes[1] if score >= 0 else classes[0]
                confidence = 1.0 / (1.0 + float(np.exp(-abs(score))))
                return label, confidence, abs(score)
            scores = scores[0]
            order = np.argsort(scores)[::-1]
            top = int(order[0])
            second = int(order[1]) if len(order) > 1 else top
            exps = np.exp(scores - np.max(scores))
            probs = exps / max(float(exps.sum()), 1e-12)
            return classes[top], float(probs[top]), float(scores[top] - scores[second])
        prediction = str(self.classifier.predict(features)[0])
        return prediction, 0.5, 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "submodel_version": self.submodel_version,
            "labels": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
            "fallback_label": LATERAL_UNCERTAIN_LABEL,
            "label_taxonomy_version": LABEL_TAXONOMY_VERSION_V241,
            "feature_names": self.feature_names or LATERAL_FEATURE_NAMES,
            "class_distance_centers": self.class_distance_centers or {},
            "class_distance_scales": self.class_distance_scales or {},
            "margin_thresholds": self.margin_thresholds or {},
            "distance_z_thresholds": self.distance_z_thresholds or {},
            "prototype_sources": self.prototype_sources,
            "prototype_subtypes": self.prototype_subtypes,
        }


def resolve_final_posture_label(parent_result: dict[str, Any], lateral_result: dict[str, Any] | None = None, temporal_state: str = "inactive") -> dict[str, Any]:
    if lateral_result is None:
        normalized = normalize_parent_lateral_display(parent_result, lateral_gate_active=False, temporal_state=temporal_state)
        return {
            "label": normalized.get("label"),
            "selected_branch": normalized.get("selected_branch", "parent"),
            "override_reason": normalized.get("override_reason", "no_lateral_resolution"),
            "fallback_reason": "",
            "lateral_fallback_requested": False,
            "normalized_parent": normalized,
        }
    lateral_label = lateral_result.get("final_display_label") or lateral_result.get("lateral_merged_label")
    fallback_requested = bool(lateral_result.get("lateral_boundary")) or lateral_label == LATERAL_UNCERTAIN_LABEL
    if fallback_requested:
        label = LATERAL_UNCERTAIN_LABEL
        selected = "lateral_temporal_hold" if temporal_state == "hold" else "lateral_fallback"
        fallback_reason = "; ".join(str(item) for item in lateral_result.get("lateral_boundary_reasons") or []) or "lateral_uncertain"
    else:
        label = lateral_label
        selected = "lateral_temporal_hold" if temporal_state == "hold" else "lateral_fine"
        fallback_reason = ""
    return {
        "label": label,
        "selected_branch": selected,
        "override_reason": "v2_4_1_lateral_merged_result_has_priority_after_gate",
        "fallback_reason": fallback_reason,
        "lateral_fallback_requested": fallback_requested,
        "normalized_parent": None,
    }


class TwoStageLateralMergedRecognizerV241:
    def __init__(self, parent_recognizer: object, lateral_model: LateralMergedFineModelV241, model_version: str = "v2_4_1_candidate", parent_model_version: str = "v2_2_candidate", lateral_hold_frames: int = 6) -> None:
        self.parent_recognizer = parent_recognizer
        self.lateral_model = lateral_model
        self.model_version = model_version
        self.parent_model_version = parent_model_version
        self.lateral_submodel_version = lateral_model.submodel_version
        self.lateral_hold_frames = max(0, int(lateral_hold_frames))
        self._last_lateral_result: dict[str, Any] | None = None
        self._last_lateral_features: dict[str, Any] | None = None
        self._missed_lateral_frames = 0
        self._candidate_lateral_frames = 0

    def reset(self) -> None:
        self._last_lateral_result = None
        self._last_lateral_features = None
        self._missed_lateral_frames = 0
        self._candidate_lateral_frames = 0
        reset = getattr(self.parent_recognizer, "reset", None)
        if callable(reset):
            reset()

    def predict_posture(self, window: np.ndarray) -> dict[str, Any]:
        parent = dict(self.parent_recognizer.predict_posture(window))
        feature_vector, feature_map = extract_lateral_features(window)
        should_run, gate_reasons = should_run_lateral_subclassifier(parent, feature_map)
        lenient_gate, lenient_reason = self._benign_parent_lateral_gate(parent, gate_reasons)
        if should_run or lenient_gate:
            self._candidate_lateral_frames += 1
        else:
            self._candidate_lateral_frames = 0
        if not should_run and lenient_gate and self._candidate_lateral_frames >= 3:
            should_run = True
            gate_reasons = list(gate_reasons) + [lenient_reason]
        payload = self._base_payload(parent, should_run, gate_reasons, feature_map)
        if bool(parent.get("subclassifier_triggered")):
            payload["lateral_gate_reason"] = "leanback_priority"
            return payload
        if should_run:
            lateral = self.lateral_model.predict_from_features(feature_vector)
            lateral["lateral_feature_summary"] = feature_map
            self._last_lateral_result = dict(lateral)
            self._last_lateral_features = dict(feature_map)
            self._missed_lateral_frames = 0
            payload.update(lateral)
            return self._apply_lateral_resolution(payload, parent, lateral, temporal_state="active")
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
                return self._apply_lateral_resolution(payload, parent, lateral, temporal_state="hold")
        if self._missed_lateral_frames > self.lateral_hold_frames:
            self._last_lateral_result = None
            self._last_lateral_features = None
            self._missed_lateral_frames = 0
        return normalize_parent_lateral_display(payload, lateral_gate_active=False, physical_features=feature_map, temporal_state="inactive")

    @staticmethod
    def _benign_parent_lateral_gate(parent: dict[str, Any], gate_reasons: list[str]) -> tuple[bool, str]:
        parent_label = parent.get("label") or parent.get("raw_label") or parent.get("parent_posture_label")
        if parent_label not in LATERAL_PARENT_LABELS:
            return False, ""
        reason_text = "; ".join(str(item) for item in gate_reasons)
        if "lateral_parent_match" not in reason_text and "lateral_prototype_match" not in reason_text:
            return False, ""
        physical_failures = []
        for reason in gate_reasons:
            for token in str(reason).split(";"):
                token = token.strip()
                if token.startswith("physical_gate_"):
                    physical_failures.append(token)
        allowed = {"physical_gate_active_area_too_low"}
        if physical_failures and set(physical_failures).issubset(allowed):
            return True, "lateral_parent_benign_physical_hold"
        return False, ""

    def _base_payload(self, parent: dict[str, Any], should_run: bool, gate_reasons: list[str], feature_map: dict[str, Any]) -> dict[str, Any]:
        payload = dict(parent)
        physical_ok, physical_reasons = lateral_physical_gate(feature_map)
        payload.update({
            "model_version": self.model_version,
            "parent_model_version": self.parent_model_version,
            "lateral_submodel_version": self.lateral_submodel_version,
            "label_taxonomy_version": LABEL_TAXONOMY_VERSION_V241,
            "lateral_subclassifier_triggered": should_run,
            "lateral_gate_candidate": any(label in ";".join(gate_reasons) for label in [STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL, DIAGONAL_SITTING_LABEL]),
            "lateral_gate_reason": "; ".join(gate_reasons) if gate_reasons else "",
            "lateral_physical_evidence_passed": bool(physical_ok),
            "lateral_physical_evidence_reasons": "; ".join(physical_reasons),
            "lateral_posture_label": None,
            "lateral_merged_label": None,
            "lateral_confidence": None,
            "lateral_margin": None,
            "lateral_boundary": False,
            "lateral_boundary_reasons": [],
            "lateral_prototype_label": None,
            "lateral_prototype_subtype": None,
            "lateral_prototype_distance": None,
            "lateral_fallback_used": False,
            "lateral_temporal_state": "active" if should_run else "inactive",
            "lateral_stable_label": None,
            "lateral_fallback_requested": False,
            "lateral_normalization_applied": False,
            "lateral_normalization_reason": "not_evaluated",
            "lateral_normalization_confidence": 0.0,
            "selected_final_branch": "parent",
            "final_override_reason": "gate_not_active",
            "final_priority_branch": "parent",
            "selected_branch": "parent",
            "override_reason": "gate_not_active",
            "fallback_reason": "",
            "parent_raw_lateral_label": parent.get("label") if parent.get("label") in LATERAL_PARENT_LABELS else None,
        })
        return payload

    def _apply_lateral_resolution(self, payload: dict[str, Any], parent: dict[str, Any], lateral: dict[str, Any], temporal_state: str) -> dict[str, Any]:
        resolution = resolve_final_posture_label(parent, lateral, temporal_state=temporal_state)
        label = resolution["label"]
        payload["label"] = label
        payload["final_display_label"] = label
        payload["posture"] = label
        payload["confidence"] = max(float(parent.get("confidence") or 0.0), LATERAL_DISPLAY_CONFIDENCE)
        payload["second_label"] = lateral.get("lateral_second_label") or parent.get("second_label")
        payload["margin"] = max(float(parent.get("margin") or 0.0), LATERAL_DISPLAY_MARGIN)
        payload["is_boundary"] = False
        payload["boundary_reasons"] = []
        payload["boundary_reason"] = None
        payload["selected_branch"] = resolution["selected_branch"]
        payload["selected_final_branch"] = resolution["selected_branch"]
        payload["final_priority_branch"] = resolution["selected_branch"]
        payload["override_reason"] = resolution["override_reason"]
        payload["final_override_reason"] = resolution["override_reason"]
        payload["fallback_reason"] = resolution["fallback_reason"]
        payload["lateral_fallback_requested"] = resolution["lateral_fallback_requested"]
        payload["lateral_temporal_state"] = temporal_state
        payload["lateral_stable_label"] = label
        payload["label_taxonomy_version"] = LABEL_TAXONOMY_VERSION_V241
        payload["parent_raw_lateral_label"] = parent.get("label") if parent.get("label") in LATERAL_PARENT_LABELS else parent.get("parent_raw_lateral_label")
        payload["lateral_normalization_applied"] = False
        payload["lateral_normalization_reason"] = "lateral_resolver_selected_final_label"
        payload["lateral_normalization_confidence"] = 0.0
        return payload


def save_lateral_merged_fine_model_v241(path: Path | str, model: LateralMergedFineModelV241) -> None:
    import joblib
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_lateral_merged_fine_model_v241(path: Path | str) -> LateralMergedFineModelV241:
    import joblib
    return joblib.load(path)


def save_lateral_merged_prototype_bank_v241(path: Path | str, model: LateralMergedFineModelV241) -> None:
    payload = {
        "labels": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
        "fallback_label": LATERAL_UNCERTAIN_LABEL,
        "label_taxonomy_version": LABEL_TAXONOMY_VERSION_V241,
        "feature_names": model.feature_names or LATERAL_FEATURE_NAMES,
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
