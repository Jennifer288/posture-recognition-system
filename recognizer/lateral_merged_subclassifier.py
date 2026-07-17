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
    _prototype_label_from_diagnosis,
    extract_lateral_features,
    lateral_physical_gate,
    should_run_lateral_subclassifier,
)

SIDE_SITTING_OR_LEANING_LABEL = "侧向坐姿"
DIAGONAL_SITTING_LABEL = ORIGINAL_DIAGONAL_SITTING_LABEL
LATERAL_UNCERTAIN_LABEL = "侧向姿势"
LABEL_TAXONOMY_VERSION = "lateral_merge_v1"
SOURCE_SUBTYPE_STANDARD_SIDE = STANDARD_SIDE_SITTING_LABEL
SOURCE_SUBTYPE_SIDE_LEANING = SIDE_LEANING_LABEL
SOURCE_SUBTYPE_DIAGONAL = DIAGONAL_SITTING_LABEL
MERGED_OUTPUT_LABELS = {SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL, LATERAL_UNCERTAIN_LABEL}
SIDE_SOURCE_LABELS = {STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL}
LATERAL_PARENT_LABELS = {STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL, DIAGONAL_SITTING_LABEL}


def merged_lateral_label(label: str | None) -> str | None:
    if label in SIDE_SOURCE_LABELS:
        return SIDE_SITTING_OR_LEANING_LABEL
    if label == DIAGONAL_SITTING_LABEL:
        return DIAGONAL_SITTING_LABEL
    if label == ORIGINAL_LATERAL_BOUNDARY_LABEL:
        return LATERAL_UNCERTAIN_LABEL
    if label in {SIDE_SITTING_OR_LEANING_LABEL, LATERAL_UNCERTAIN_LABEL}:
        # ``侧向坐姿`` in legacy V2.3 meant coarse fallback. V2.4 emits the same
        # text only after its own resolver has chosen the merged formal class.
        return LATERAL_UNCERTAIN_LABEL if label == ORIGINAL_LATERAL_BOUNDARY_LABEL else label
    return label


def source_subtype_for_label(label: str | None) -> str | None:
    if label in {STANDARD_SIDE_SITTING_LABEL, SIDE_LEANING_LABEL, DIAGONAL_SITTING_LABEL}:
        return label
    return None


def normalize_parent_lateral_display(parent_result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(parent_result)
    parent_label = payload.get("label") or payload.get("final_display_label")
    if parent_label in SIDE_SOURCE_LABELS:
        payload["parent_raw_lateral_label"] = parent_label
        payload["label"] = SIDE_SITTING_OR_LEANING_LABEL
        payload["final_display_label"] = SIDE_SITTING_OR_LEANING_LABEL
        payload["label_taxonomy_version"] = LABEL_TAXONOMY_VERSION
        payload.setdefault("selected_branch", "parent_lateral_label_normalized")
        payload.setdefault("final_priority_branch", "parent_lateral_label_normalized")
        payload.setdefault("override_reason", "v2_4_label_merge_normalized_parent_side_label")
    elif parent_label == DIAGONAL_SITTING_LABEL:
        payload["parent_raw_lateral_label"] = parent_label
        payload["label_taxonomy_version"] = LABEL_TAXONOMY_VERSION
    return payload


@dataclass
class LateralMergedFineModel:
    prototypes: dict[str, list[np.ndarray]]
    prototype_sources: dict[str, list[str]]
    prototype_subtypes: dict[str, list[str]]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    margin_threshold: float = 0.10
    confidence_threshold: float = 0.50
    distance_thresholds: dict[str, float] | None = None
    pair_margin_thresholds: dict[str, float] | None = None
    classifier: object | None = None
    feature_names: list[str] | None = None
    submodel_version: str = "lateral_merged_subclassifier_v2_4_candidate"

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
            raise ValueError("LateralMergedFineModel requires at least one prototype")
        best_label, best_source, best_subtype, best_distance = ordered[0]
        second_label, second_source, second_subtype, second_distance = self._second_other_label(ordered, best_label)
        prototype_margin = float(second_distance - best_distance)
        prototype_confidence = float(1.0 / (1.0 + best_distance))
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
        threshold = float((self.distance_thresholds or {}).get(best_label, np.inf))
        if best_distance > threshold:
            reasons.append("out_of_distribution")
        if prototype_margin < self.margin_threshold:
            reasons.append("low_prototype_margin")
        if classifier_margin is not None and classifier_margin < self.margin_threshold:
            reasons.append("low_classifier_margin")
        if classifier_confidence is not None and classifier_confidence < self.confidence_threshold:
            reasons.append("low_classifier_confidence")
        pair_key = "::".join(sorted([best_label, second_label]))
        pair_threshold = float((self.pair_margin_thresholds or {}).get(pair_key, self.margin_threshold))
        if {best_label, second_label} == {SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL} and prototype_margin < pair_threshold:
            reasons.append("side_vs_diagonal_overlap")
        if resolved_label not in {SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL, LATERAL_UNCERTAIN_LABEL}:
            resolved_label = LATERAL_UNCERTAIN_LABEL
            reasons.append("unknown_lateral_label")

        lateral_boundary = bool(reasons) or resolved_label == LATERAL_UNCERTAIN_LABEL
        final_label = LATERAL_UNCERTAIN_LABEL if lateral_boundary else resolved_label
        confidence = classifier_confidence if classifier_confidence is not None else prototype_confidence
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
            "lateral_second_label": second_label,
            "lateral_second_subtype": second_subtype,
            "lateral_second_prototype_source": second_source,
            "lateral_second_distance": round(float(second_distance), 6),
            "lateral_prototype_margin": round(float(prototype_margin), 6),
            "lateral_fallback_used": final_label == LATERAL_UNCERTAIN_LABEL,
            "lateral_out_of_distribution": "out_of_distribution" in reasons,
            "lateral_classifier_label": classifier_label,
            "label_taxonomy_version": LABEL_TAXONOMY_VERSION,
        }

    def _prototype_distances(self, scaled: np.ndarray) -> list[tuple[str, str, str, float]]:
        rows: list[tuple[str, str, str, float]] = []
        for label, protos in self.prototypes.items():
            sources = self.prototype_sources.get(label, [])
            subtypes = self.prototype_subtypes.get(label, [])
            for idx, proto in enumerate(protos):
                source = sources[idx] if idx < len(sources) else f"{label}::prototype_{idx}"
                subtype = subtypes[idx] if idx < len(subtypes) else label
                rows.append((label, source, subtype, float(np.linalg.norm(scaled - self._scale(proto)))))
        return sorted(rows, key=lambda item: item[3])

    def _scale(self, vector: np.ndarray) -> np.ndarray:
        scale = np.where(np.abs(self.feature_scale) < 1e-9, 1.0, self.feature_scale)
        return (np.asarray(vector, dtype=float) - self.feature_mean) / scale

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
            "label_taxonomy_version": LABEL_TAXONOMY_VERSION,
            "original_to_merged_label_mapping": {
                STANDARD_SIDE_SITTING_LABEL: SIDE_SITTING_OR_LEANING_LABEL,
                SIDE_LEANING_LABEL: SIDE_SITTING_OR_LEANING_LABEL,
                DIAGONAL_SITTING_LABEL: DIAGONAL_SITTING_LABEL,
            },
            "feature_names": self.feature_names or LATERAL_FEATURE_NAMES,
            "margin_threshold": self.margin_threshold,
            "confidence_threshold": self.confidence_threshold,
            "distance_thresholds": self.distance_thresholds or {},
            "pair_margin_thresholds": self.pair_margin_thresholds or {},
            "prototype_sources": self.prototype_sources,
            "prototype_subtypes": self.prototype_subtypes,
        }


def resolve_final_posture_label(
    parent_result: dict[str, Any],
    lateral_result: dict[str, Any] | None = None,
    temporal_state: str = "inactive",
) -> dict[str, Any]:
    if lateral_result is None:
        normalized = normalize_parent_lateral_display(parent_result)
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
        "override_reason": "v2_4_lateral_merged_result_has_priority_after_gate",
        "fallback_reason": fallback_reason,
        "lateral_fallback_requested": fallback_requested,
        "normalized_parent": None,
    }


class TwoStageLateralMergedRecognizer:
    def __init__(
        self,
        parent_recognizer: object,
        lateral_model: LateralMergedFineModel,
        model_version: str = "v2_4_candidate",
        parent_model_version: str = "v2_2_candidate",
        lateral_hold_frames: int = 6,
    ) -> None:
        self.parent_recognizer = parent_recognizer
        self.lateral_model = lateral_model
        self.model_version = model_version
        self.parent_model_version = parent_model_version
        self.lateral_submodel_version = lateral_model.submodel_version
        self.lateral_hold_frames = max(0, int(lateral_hold_frames))
        self._last_lateral_result: dict[str, Any] | None = None
        self._missed_lateral_frames = 0

    def reset(self) -> None:
        self._last_lateral_result = None
        self._missed_lateral_frames = 0
        reset = getattr(self.parent_recognizer, "reset", None)
        if callable(reset):
            reset()

    def predict_posture(self, window: np.ndarray) -> dict[str, Any]:
        parent = dict(self.parent_recognizer.predict_posture(window))
        feature_vector, feature_map = extract_lateral_features(window)
        should_run, gate_reasons = should_run_lateral_subclassifier(parent, feature_map)
        payload = self._base_payload(parent, should_run, gate_reasons)
        if bool(parent.get("subclassifier_triggered")):
            payload["lateral_gate_reason"] = "leanback_priority"
            return payload
        if should_run:
            lateral = self.lateral_model.predict_from_features(feature_vector)
            lateral["lateral_feature_summary"] = feature_map
            self._last_lateral_result = dict(lateral)
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
            self._missed_lateral_frames = 0
        return normalize_parent_lateral_display(payload)

    def _base_payload(self, parent: dict[str, Any], should_run: bool, gate_reasons: list[str]) -> dict[str, Any]:
        payload = dict(parent)
        payload.update({
            "model_version": self.model_version,
            "parent_model_version": self.parent_model_version,
            "lateral_submodel_version": self.lateral_submodel_version,
            "label_taxonomy_version": LABEL_TAXONOMY_VERSION,
            "lateral_subclassifier_triggered": should_run,
            "lateral_gate_reason": "; ".join(gate_reasons) if gate_reasons else "",
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
            "final_priority_branch": "parent",
            "selected_branch": "parent",
            "override_reason": "gate_not_active",
            "fallback_reason": "",
            "parent_raw_lateral_label": parent.get("label") if parent.get("label") in LATERAL_PARENT_LABELS else None,
        })
        return normalize_parent_lateral_display(payload) if not should_run else payload

    def _apply_lateral_resolution(
        self,
        payload: dict[str, Any],
        parent: dict[str, Any],
        lateral: dict[str, Any],
        temporal_state: str,
    ) -> dict[str, Any]:
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
        payload["final_priority_branch"] = resolution["selected_branch"]
        payload["override_reason"] = resolution["override_reason"]
        payload["fallback_reason"] = resolution["fallback_reason"]
        payload["lateral_fallback_requested"] = resolution["lateral_fallback_requested"]
        payload["lateral_temporal_state"] = temporal_state
        payload["lateral_stable_label"] = label
        payload["label_taxonomy_version"] = LABEL_TAXONOMY_VERSION
        payload["parent_raw_lateral_label"] = parent.get("label") if parent.get("label") in LATERAL_PARENT_LABELS else parent.get("parent_raw_lateral_label")
        return payload


def save_lateral_merged_fine_model(path: Path | str, model: LateralMergedFineModel) -> None:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_lateral_merged_fine_model(path: Path | str) -> LateralMergedFineModel:
    import joblib

    return joblib.load(path)


def save_lateral_merged_prototype_bank(path: Path | str, model: LateralMergedFineModel) -> None:
    payload = {
        "labels": [SIDE_SITTING_OR_LEANING_LABEL, DIAGONAL_SITTING_LABEL],
        "fallback_label": LATERAL_UNCERTAIN_LABEL,
        "label_taxonomy_version": LABEL_TAXONOMY_VERSION,
        "feature_names": model.feature_names or LATERAL_FEATURE_NAMES,
        "prototypes": {label: [proto.tolist() for proto in protos] for label, protos in model.prototypes.items()},
        "prototype_sources": model.prototype_sources,
        "prototype_subtypes": model.prototype_subtypes,
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "distance_thresholds": model.distance_thresholds or {},
        "pair_margin_thresholds": model.pair_margin_thresholds or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
