from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .feature_extractor import as_frames, window_average


PARENT_LEANBACK_LABEL = "后靠/瘫坐类"
FINE_LEANBACK_LABEL = "后仰靠背坐"
FINE_SLOUCH_LABEL = "后靠/瘫坐类"
FINE_BOUNDARY_LABEL = "后靠坐姿"
LEANBACK_RELATED_LABELS = {PARENT_LEANBACK_LABEL, FINE_LEANBACK_LABEL, FINE_SLOUCH_LABEL}


def should_run_leanback_subclassifier(stage1_result: dict[str, Any], features: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """Return whether the leanback fine classifier should run.

    The gate is intentionally conservative: only parent labels, second labels,
    raw labels, or prototype diagnoses that explicitly point at the leanback
    family can activate the fine classifier. Other postures and non-human
    occupancy states must stay on the V2.1 path.
    """

    reasons: list[str] = []
    candidates = [
        stage1_result.get("label"),
        stage1_result.get("raw_label"),
    ]
    diagnosis = stage1_result.get("prototype_diagnosis")
    if isinstance(diagnosis, dict):
        candidates.append(diagnosis.get("label"))
    for candidate in candidates:
        if candidate in LEANBACK_RELATED_LABELS:
            reasons.append(f"candidate={candidate}")
    if not reasons:
        return False, reasons
    if features is None:
        return True, reasons
    physical_ok, physical_reasons = leanback_physical_gate(features)
    reasons.extend(physical_reasons)
    return physical_ok, reasons


def leanback_physical_gate(features: dict[str, Any]) -> tuple[bool, list[str]]:
    """Conservative physical gate learned from the H1/H2 separability study."""

    reasons: list[str] = []
    left_share = float(features.get("left_share", 0.5))
    row_0_3 = float(features.get("row_0_3_share", 0.0))
    row_4_7 = float(features.get("row_4_7_share", 0.0))
    cop_y = float(features.get("cop_y", 0.0))
    if not 0.40 <= left_share <= 0.62:
        reasons.append("physical_gate_left_right_out_of_range")
    if row_0_3 + row_4_7 < 0.72:
        reasons.append("physical_gate_front_middle_support_too_low")
    if not 2.40 <= cop_y <= 5.80:
        reasons.append("physical_gate_cop_y_out_of_range")
    if reasons:
        return False, reasons
    return True, ["physical_gate_pass"]


def extract_leanback_features(window: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Extract compact physical features for leanback/slouch subdivision.

    Rows 0-3 are treated as the front edge, matching the existing diagnostics.
    Total pressure is included as an auxiliary feature; normalized shape and
    front/back landing features carry the main decision signal.
    """

    frames = as_frames(window)
    mean = window_average(frames)
    total = float(mean.sum())
    safe_total = total if total > 1e-9 else 1.0
    normalized = mean / safe_total
    rows = [float(mean[start:end, :].sum() / safe_total) for start, end in [(0, 4), (4, 8), (8, 12), (12, 16)]]
    front = float(mean[:8, :].sum() / safe_total)
    back = float(mean[8:, :].sum() / safe_total)
    left = float(mean[:, :8].sum() / safe_total)
    right = float(mean[:, 8:].sum() / safe_total)
    x = np.arange(16, dtype=float).reshape(1, 16)
    y = np.arange(16, dtype=float).reshape(16, 1)
    cop_x = float((normalized * x).sum())
    cop_y = float((normalized * y).sum())
    active_mask = mean > 15.0
    active_points = int(active_mask.sum())
    bbox = _bbox(active_mask)
    bbox_height = float(max(0, bbox[2] - bbox[0] + 1)) if bbox is not None else 0.0
    bbox_width = float(max(0, bbox[3] - bbox[1] + 1)) if bbox is not None else 0.0
    peak_share = float(mean.max() / safe_total) if mean.size else 0.0
    hhi = float((normalized**2).sum())
    totals = frames.sum(axis=(1, 2))
    frame_safe = np.where(totals > 1e-9, totals, 1.0)
    frame_norm = frames / frame_safe.reshape(-1, 1, 1)
    frame_cop_y = (frame_norm * y.reshape(1, 16, 1)).sum(axis=(1, 2))
    cop_y_std = float(np.nanstd(frame_cop_y)) if len(frame_cop_y) else 0.0
    total_cv = float(np.std(totals) / max(abs(float(np.mean(totals))), 1e-9)) if len(totals) else 0.0
    largest_region, region_count = _connected_regions(active_mask)

    feature_map = {
        "cop_x": cop_x,
        "cop_y": cop_y,
        "front_share": front,
        "back_share": back,
        "front_back_balance": front - back,
        "row_0_3_share": rows[0],
        "row_4_7_share": rows[1],
        "row_8_11_share": rows[2],
        "row_12_15_share": rows[3],
        "left_share": left,
        "right_share": right,
        "left_right_balance": left - right,
        "active_area_ratio": active_points / 256.0,
        "peak_share": peak_share,
        "hhi": hhi,
        "bbox_height": bbox_height / 16.0,
        "bbox_width": bbox_width / 16.0,
        "largest_region_ratio": largest_region / 256.0,
        "region_count_scaled": region_count / 10.0,
        "cop_y_std": cop_y_std / 15.0,
        "total_cv": total_cv,
        "log_total": float(np.log1p(total) / 10.0),
    }
    vector = np.asarray([feature_map[key] for key in LEANBACK_FEATURE_NAMES], dtype=float)
    return vector, feature_map


LEANBACK_FEATURE_NAMES = [
    "cop_x",
    "cop_y",
    "front_share",
    "back_share",
    "front_back_balance",
    "row_0_3_share",
    "row_4_7_share",
    "row_8_11_share",
    "row_12_15_share",
    "left_share",
    "right_share",
    "left_right_balance",
    "active_area_ratio",
    "peak_share",
    "hhi",
    "bbox_height",
    "bbox_width",
    "largest_region_ratio",
    "region_count_scaled",
    "cop_y_std",
    "total_cv",
    "log_total",
]


@dataclass
class LeanbackFineModel:
    prototypes: dict[str, np.ndarray]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    margin_threshold: float = 0.08
    confidence_threshold: float = 0.55
    distance_thresholds: dict[str, float] | None = None
    classifier: object | None = None
    feature_names: list[str] | None = None
    submodel_version: str = "leanback_subclassifier_v2_2_candidate"

    def predict(self, window: np.ndarray) -> dict[str, Any]:
        features, feature_map = extract_leanback_features(window)
        result = self.predict_from_features(features)
        result["fine_feature_summary"] = feature_map
        return result

    def predict_from_features(self, features: np.ndarray) -> dict[str, Any]:
        vector = np.asarray(features, dtype=float).reshape(-1)
        scaled = self._scale(vector)
        distances = {
            label: float(np.linalg.norm(scaled - self._scale(proto)))
            for label, proto in self.prototypes.items()
        }
        ordered = sorted(distances.items(), key=lambda item: item[1])
        best_label, best_distance = ordered[0]
        second_label, second_distance = ordered[1] if len(ordered) > 1 else ("", best_distance + 1.0)
        prototype_margin = float(second_distance - best_distance)
        confidence = float(1.0 / (1.0 + best_distance))
        classifier_label = None
        classifier_margin = None
        classifier_confidence = None
        if self.classifier is not None:
            classifier_label, classifier_confidence, classifier_margin = self._classifier_prediction(scaled.reshape(1, -1))

        resolved_label = str(classifier_label or best_label)
        if classifier_label is not None and classifier_label != best_label and prototype_margin < self.margin_threshold * 1.5:
            resolved_label = FINE_BOUNDARY_LABEL
        reasons: list[str] = []
        thresholds = self.distance_thresholds or {}
        threshold = float(thresholds.get(best_label, np.inf))
        if prototype_margin < self.margin_threshold:
            reasons.append("prototype_boundary")
        if best_distance > threshold:
            reasons.append("out_of_distribution_boundary")
        if classifier_margin is not None and classifier_margin < self.margin_threshold:
            reasons.append("classifier_boundary")
        if classifier_confidence is not None and classifier_confidence < self.confidence_threshold:
            reasons.append("classifier_boundary")
        if resolved_label == FINE_BOUNDARY_LABEL:
            reasons.append("classifier_prototype_conflict")

        fine_boundary = bool(reasons)
        final_label = FINE_BOUNDARY_LABEL if fine_boundary else resolved_label
        if final_label not in {FINE_LEANBACK_LABEL, FINE_SLOUCH_LABEL, FINE_BOUNDARY_LABEL}:
            final_label = FINE_BOUNDARY_LABEL
            fine_boundary = True
            reasons.append("unknown_fine_label")

        return {
            "fine_posture_label": final_label,
            "final_display_label": final_label,
            "fine_confidence": round(float(classifier_confidence if classifier_confidence is not None else confidence), 6),
            "fine_margin": round(float(classifier_margin if classifier_margin is not None else prototype_margin), 6),
            "fine_boundary": fine_boundary,
            "fine_boundary_reasons": sorted(set(reasons)),
            "fine_prototype_label": best_label,
            "fine_prototype_distance": round(best_distance, 6),
            "fine_second_label": second_label,
            "fine_second_distance": round(second_distance, 6),
            "fallback_used": final_label == FINE_BOUNDARY_LABEL,
        }

    def _scale(self, vector: np.ndarray) -> np.ndarray:
        scale = np.where(np.abs(self.feature_scale) < 1e-9, 1.0, self.feature_scale)
        return (np.asarray(vector, dtype=float) - self.feature_mean) / scale

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
            if scores.ndim == 1:
                score = float(scores[0])
                classes = [str(item) for item in self.classifier.classes_]
                label = classes[1] if score >= 0 else classes[0]
                confidence = 1.0 / (1.0 + float(np.exp(-abs(score))))
                return label, confidence, abs(score)
        prediction = str(self.classifier.predict(features)[0])
        return prediction, 0.5, 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "submodel_version": self.submodel_version,
            "feature_names": self.feature_names or LEANBACK_FEATURE_NAMES,
            "labels": [FINE_LEANBACK_LABEL, FINE_SLOUCH_LABEL],
            "fallback_label": FINE_BOUNDARY_LABEL,
            "margin_threshold": self.margin_threshold,
            "confidence_threshold": self.confidence_threshold,
            "distance_thresholds": self.distance_thresholds or {},
        }


class TwoStageLeanbackRecognizer:
    def __init__(
        self,
        parent_recognizer: object,
        fine_model: LeanbackFineModel,
        model_version: str = "v2_2_candidate",
        parent_model_version: str = "v2_1_candidate",
    ) -> None:
        self.parent_recognizer = parent_recognizer
        self.fine_model = fine_model
        self.model_version = model_version
        self.parent_model_version = parent_model_version
        self.submodel_version = fine_model.submodel_version

    def predict_posture(self, window: np.ndarray) -> dict[str, Any]:
        parent = dict(self.parent_recognizer.predict_posture(window))
        feature_vector, feature_map = extract_leanback_features(window)
        should_run, gate_reasons = should_run_leanback_subclassifier(parent, feature_map)
        payload = dict(parent)
        payload.update(
            {
                "parent_posture_label": parent.get("label"),
                "fine_posture_label": None,
                "final_display_label": parent.get("label"),
                "subclassifier_triggered": should_run,
                "subclassifier_gate_reason": "; ".join(gate_reasons) if gate_reasons else "",
                "fine_confidence": None,
                "fine_margin": None,
                "fine_boundary": False,
                "fine_boundary_reasons": [],
                "fine_prototype_label": None,
                "fine_prototype_distance": None,
                "fallback_used": False,
                "model_version": self.model_version,
                "parent_model_version": self.parent_model_version,
                "submodel_version": self.submodel_version,
            }
        )
        if not should_run:
            return payload

        fine = self.fine_model.predict_from_features(feature_vector)
        fine["fine_feature_summary"] = feature_map
        payload.update(fine)
        payload["label"] = fine["final_display_label"]
        payload["confidence"] = parent.get("confidence", fine["fine_confidence"])
        payload["second_label"] = fine.get("fine_second_label") or parent.get("second_label")
        payload["margin"] = parent.get("margin", fine["fine_margin"])
        # A fine boundary falls back to the coarse "后靠坐姿" label. Keep the
        # top-level boundary false so SeatAnalyzer/Smoother do not hide the
        # safe fallback posture from the GUI.
        payload["is_boundary"] = bool(parent.get("is_boundary", False)) and not fine.get("fallback_used", False)
        if fine.get("fallback_used"):
            payload["is_boundary"] = False
        payload["boundary_reasons"] = list(parent.get("boundary_reasons") or [])
        payload["fine_boundary_reasons"] = list(fine.get("fine_boundary_reasons") or [])
        return payload


def save_leanback_fine_model(path: Path | str, model: LeanbackFineModel) -> None:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_leanback_fine_model(path: Path | str) -> LeanbackFineModel:
    import joblib

    return joblib.load(path)


def save_leanback_prototype_bank(path: Path | str, model: LeanbackFineModel) -> None:
    payload = {
        "labels": [FINE_LEANBACK_LABEL, FINE_SLOUCH_LABEL],
        "fallback_label": FINE_BOUNDARY_LABEL,
        "feature_names": model.feature_names or LEANBACK_FEATURE_NAMES,
        "prototypes": {label: vector.tolist() for label, vector in model.prototypes.items()},
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "distance_thresholds": model.distance_thresholds or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if len(points) == 0:
        return None
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_regions(mask: np.ndarray) -> tuple[int, int]:
    seen = np.zeros(mask.shape, dtype=bool)
    largest = 0
    count = 0
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for nr in range(max(0, cr - 1), min(rows, cr + 2)):
                    for nc in range(max(0, cc - 1), min(cols, cc + 2)):
                        if mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            largest = max(largest, size)
    return largest, count
