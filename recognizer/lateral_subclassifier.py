from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .feature_extractor import as_frames, window_average


STANDARD_SIDE_SITTING_LABEL = "标准侧坐"
DIAGONAL_SITTING_LABEL = "斜跨坐"
SIDE_LEANING_LABEL = "侧身倚靠坐"
LATERAL_BOUNDARY_LABEL = "侧向坐姿"
LATERAL_RELATED_LABELS = {STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL, LATERAL_BOUNDARY_LABEL}


LATERAL_FEATURE_NAMES = [
    "cop_x",
    "cop_y",
    "left_share",
    "right_share",
    "left_right_balance",
    "left_right_abs_balance",
    "front_share",
    "back_share",
    "front_back_balance",
    "row_0_3_share",
    "row_4_7_share",
    "row_8_11_share",
    "row_12_15_share",
    "col_0_3_share",
    "col_4_7_share",
    "col_8_11_share",
    "col_12_15_share",
    "front_left_share",
    "front_right_share",
    "back_left_share",
    "back_right_share",
    "active_area_ratio",
    "peak_share",
    "hhi",
    "bbox_row_min",
    "bbox_col_min",
    "bbox_row_max",
    "bbox_col_max",
    "bbox_height",
    "bbox_width",
    "bbox_center_x",
    "bbox_center_y",
    "largest_region_ratio",
    "region_count_scaled",
    "max_row",
    "max_col",
    "cop_x_std",
    "cop_y_std",
    "left_right_std",
    "front_back_std",
    "total_cv",
    "log_total",
]


def should_run_lateral_subclassifier(stage1_result: dict[str, Any], features: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """Return whether the V2.3 lateral local resolver should run.

    V2.2 leanback has priority. For non-leanback postures, the gate requires
    either a lateral parent label, or a parent Boundary region with lateral RF,
    Prototype, or physical evidence. This keeps ordinary upright/front/back
    postures out of the local resolver even when a transient top-2 candidate is
    lateral.
    """

    if bool(stage1_result.get("subclassifier_triggered")):
        return False, ["gate_rejected_leanback_priority"]

    reasons: list[str] = []
    primary_candidates = [stage1_result.get("label"), stage1_result.get("final_display_label"), stage1_result.get("parent_posture_label")]
    primary_lateral = [item for item in primary_candidates if item in {STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL}]
    parent_boundary = bool(stage1_result.get("is_boundary")) or stage1_result.get("label") in {None, "", "边界/不确定"}
    parent_label = stage1_result.get("label") or stage1_result.get("final_display_label") or stage1_result.get("parent_posture_label")

    if primary_lateral:
        reasons.extend(f"lateral_parent_match={item}" for item in primary_lateral)
        if features is not None:
            physical_ok, physical_reasons = lateral_physical_gate(features)
            if physical_ok:
                extra = ["lateral_boundary_region"] if parent_boundary else []
                return True, reasons + physical_reasons + extra
            return False, reasons + physical_reasons
        if not parent_boundary:
            return True, reasons
        return True, reasons + ["lateral_boundary_region"]

    if parent_label not in {None, "", "边界/不确定"} and not parent_boundary:
        return False, [f"gate_rejected_parent_non_lateral={parent_label}"]

    secondary_candidates = [stage1_result.get("raw_label"), stage1_result.get("second_label")]
    diagnosis = stage1_result.get("prototype_diagnosis")
    if isinstance(diagnosis, dict):
        secondary_candidates.append(diagnosis.get("label"))
    for candidate in secondary_candidates:
        if candidate in {STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL}:
            reasons.append(f"lateral_candidate={candidate}")
    if parent_boundary and reasons:
        if features is not None:
            physical_ok, physical_reasons = lateral_physical_gate(features)
            if physical_ok:
                return True, reasons + physical_reasons + ["lateral_boundary_region"]
            return False, reasons + physical_reasons
        reasons.append("lateral_boundary_region")
        return True, reasons

    if features is not None and parent_boundary:
        physical_ok, physical_reasons = lateral_physical_gate(features)
        if physical_ok:
            return True, physical_reasons + ["lateral_boundary_region"]
        return False, physical_reasons

    return False, reasons or ["gate_rejected_no_lateral_evidence"]


def lateral_physical_gate(features: dict[str, Any]) -> tuple[bool, list[str]]:
    """Conservative seat-only physical gate for lateral postures."""

    left = float(features.get("left_share", 0.5))
    right = float(features.get("right_share", 0.5))
    active_area = float(features.get("active_area_ratio", 0.0))
    front = float(features.get("front_share", 0.5))
    back = float(features.get("back_share", 0.5))
    lr_abs = abs(left - right)
    reasons: list[str] = []
    if active_area < 0.22:
        reasons.append("physical_gate_active_area_too_low")
    if lr_abs < 0.10:
        reasons.append("physical_gate_left_right_balance_too_small")
    if front < 0.18 or back < 0.28:
        reasons.append("physical_gate_front_back_support_out_of_range")
    row_8_11 = float(features.get("row_8_11_share", 0.0))
    cop_y = float(features.get("cop_y", 0.0))
    if float(features.get("row_4_7_share", 0.0)) > 0.50 and row_8_11 < 0.25:
        reasons.append("physical_gate_front_middle_dominant_without_back_support")
    if row_8_11 < 0.26 and cop_y < 6.10:
        reasons.append("physical_gate_missing_lateral_back_mid_extension")
    if reasons:
        return False, reasons
    return True, ["lateral_physical_gate_pass"]


def extract_lateral_features(window: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Extract compact, seat-observable features for lateral subdivision."""

    frames = as_frames(window)
    mean = window_average(frames)
    total = float(mean.sum())
    safe_total = total if total > 1e-9 else 1.0
    normalized = mean / safe_total
    rows = [float(mean[start:end, :].sum() / safe_total) for start, end in [(0, 4), (4, 8), (8, 12), (12, 16)]]
    cols = [float(mean[:, start:end].sum() / safe_total) for start, end in [(0, 4), (4, 8), (8, 12), (12, 16)]]
    front = float(mean[:8, :].sum() / safe_total)
    back = float(mean[8:, :].sum() / safe_total)
    left = float(mean[:, :8].sum() / safe_total)
    right = float(mean[:, 8:].sum() / safe_total)
    quadrants = [
        float(mean[:8, :8].sum() / safe_total),
        float(mean[:8, 8:].sum() / safe_total),
        float(mean[8:, :8].sum() / safe_total),
        float(mean[8:, 8:].sum() / safe_total),
    ]
    x = np.arange(16, dtype=float).reshape(1, 16)
    y = np.arange(16, dtype=float).reshape(16, 1)
    cop_x = float((normalized * x).sum())
    cop_y = float((normalized * y).sum())
    active_mask = mean > 15.0
    active_points = int(active_mask.sum())
    bbox = _bbox(active_mask)
    if bbox is None:
        r0 = c0 = r1 = c1 = 0.0
        bbox_height = bbox_width = center_x = center_y = 0.0
    else:
        ir0, ic0, ir1, ic1 = bbox
        r0, c0, r1, c1 = float(ir0) / 15.0, float(ic0) / 15.0, float(ir1) / 15.0, float(ic1) / 15.0
        bbox_height = float(max(0, ir1 - ir0 + 1)) / 16.0
        bbox_width = float(max(0, ic1 - ic0 + 1)) / 16.0
        center_y = float((ir0 + ir1) / 2.0) / 15.0
        center_x = float((ic0 + ic1) / 2.0) / 15.0
    max_index = np.unravel_index(int(np.argmax(mean)), mean.shape) if mean.size else (0, 0)
    peak_share = float(mean.max() / safe_total) if mean.size else 0.0
    hhi = float((normalized**2).sum())
    largest_region, region_count = _connected_regions(active_mask)

    totals = frames.sum(axis=(1, 2))
    frame_safe = np.where(totals > 1e-9, totals, 1.0)
    frame_norm = frames / frame_safe.reshape(-1, 1, 1)
    fx = x.reshape(1, 1, 16)
    fy = y.reshape(1, 16, 1)
    frame_cop_x = (frame_norm * fx).sum(axis=(1, 2))
    frame_cop_y = (frame_norm * fy).sum(axis=(1, 2))
    frame_left = frames[:, :, :8].sum(axis=(1, 2)) / frame_safe
    frame_right = frames[:, :, 8:].sum(axis=(1, 2)) / frame_safe
    frame_front = frames[:, :8, :].sum(axis=(1, 2)) / frame_safe
    frame_back = frames[:, 8:, :].sum(axis=(1, 2)) / frame_safe

    feature_map = {
        "cop_x": cop_x,
        "cop_y": cop_y,
        "left_share": left,
        "right_share": right,
        "left_right_balance": left - right,
        "left_right_abs_balance": abs(left - right),
        "front_share": front,
        "back_share": back,
        "front_back_balance": front - back,
        "row_0_3_share": rows[0],
        "row_4_7_share": rows[1],
        "row_8_11_share": rows[2],
        "row_12_15_share": rows[3],
        "col_0_3_share": cols[0],
        "col_4_7_share": cols[1],
        "col_8_11_share": cols[2],
        "col_12_15_share": cols[3],
        "front_left_share": quadrants[0],
        "front_right_share": quadrants[1],
        "back_left_share": quadrants[2],
        "back_right_share": quadrants[3],
        "active_area_ratio": active_points / 256.0,
        "peak_share": peak_share,
        "hhi": hhi,
        "bbox_row_min": r0,
        "bbox_col_min": c0,
        "bbox_row_max": r1,
        "bbox_col_max": c1,
        "bbox_height": bbox_height,
        "bbox_width": bbox_width,
        "bbox_center_x": center_x,
        "bbox_center_y": center_y,
        "largest_region_ratio": largest_region / 256.0,
        "region_count_scaled": region_count / 10.0,
        "max_row": float(max_index[0]) / 15.0,
        "max_col": float(max_index[1]) / 15.0,
        "cop_x_std": float(np.nanstd(frame_cop_x)) / 15.0 if len(frame_cop_x) else 0.0,
        "cop_y_std": float(np.nanstd(frame_cop_y)) / 15.0 if len(frame_cop_y) else 0.0,
        "left_right_std": float(np.nanstd(frame_left - frame_right)) if len(frame_left) else 0.0,
        "front_back_std": float(np.nanstd(frame_front - frame_back)) if len(frame_front) else 0.0,
        "total_cv": float(np.std(totals) / max(abs(float(np.mean(totals))), 1e-9)) if len(totals) else 0.0,
        "log_total": float(np.log1p(total) / 10.0),
    }
    vector = np.asarray([feature_map[name] for name in LATERAL_FEATURE_NAMES], dtype=float)
    return vector, feature_map


@dataclass
class LateralFineModel:
    prototypes: dict[str, list[np.ndarray]]
    prototype_sources: dict[str, list[str]]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    margin_threshold: float = 0.10
    confidence_threshold: float = 0.50
    distance_thresholds: dict[str, float] | None = None
    pair_margin_thresholds: dict[str, float] | None = None
    classifier: object | None = None
    feature_names: list[str] | None = None
    submodel_version: str = "lateral_subclassifier_v2_3_candidate"

    def predict(self, window: np.ndarray) -> dict[str, Any]:
        features, feature_map = extract_lateral_features(window)
        result = self.predict_from_features(features)
        result["lateral_feature_summary"] = feature_map
        return result

    def predict_from_features(self, features: np.ndarray) -> dict[str, Any]:
        vector = np.asarray(features, dtype=float).reshape(-1)
        scaled = self._scale(vector)
        all_distances: list[tuple[str, str, float]] = []
        for label, protos in self.prototypes.items():
            sources = self.prototype_sources.get(label, [])
            for idx, proto in enumerate(protos):
                source = sources[idx] if idx < len(sources) else f"{label}::prototype_{idx}"
                all_distances.append((label, source, float(np.linalg.norm(scaled - self._scale(proto)))))
        if not all_distances:
            raise ValueError("LateralFineModel requires at least one prototype")
        ordered = sorted(all_distances, key=lambda item: item[2])
        best_label, best_source, best_distance = ordered[0]
        second_label, second_source, second_distance = self._second_other_label(ordered, best_label)
        prototype_margin = float(second_distance - best_distance)
        confidence = float(1.0 / (1.0 + best_distance))
        classifier_label = None
        classifier_confidence = None
        classifier_margin = None
        if self.classifier is not None:
            classifier_label, classifier_confidence, classifier_margin = self._classifier_prediction(scaled.reshape(1, -1))

        resolved_label = str(classifier_label or best_label)
        reasons: list[str] = []
        if classifier_label is not None and classifier_label != best_label:
            resolved_label = LATERAL_BOUNDARY_LABEL
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
        if {best_label, second_label} == {SIDE_LEANING_LABEL, DIAGONAL_SITTING_LABEL} and prototype_margin < pair_threshold:
            reasons.append("side_leaning_diagonal_overlap")
        if {best_label, second_label} == {SIDE_LEANING_LABEL, STANDARD_SIDE_SITTING_LABEL} and prototype_margin < pair_threshold:
            reasons.append("side_leaning_standard_side_overlap")
        if resolved_label not in {STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL, LATERAL_BOUNDARY_LABEL}:
            resolved_label = LATERAL_BOUNDARY_LABEL
            reasons.append("unknown_lateral_label")

        lateral_boundary = bool(reasons) or resolved_label == LATERAL_BOUNDARY_LABEL
        final_label = LATERAL_BOUNDARY_LABEL if lateral_boundary else resolved_label
        return {
            "lateral_posture_label": final_label,
            "final_display_label": final_label,
            "lateral_confidence": round(float(classifier_confidence if classifier_confidence is not None else confidence), 6),
            "lateral_margin": round(float(classifier_margin if classifier_margin is not None else prototype_margin), 6),
            "lateral_boundary": lateral_boundary,
            "lateral_boundary_reasons": sorted(set(reasons)),
            "lateral_prototype_label": best_label,
            "lateral_prototype_source": best_source,
            "lateral_prototype_distance": round(float(best_distance), 6),
            "lateral_second_label": second_label,
            "lateral_second_prototype_source": second_source,
            "lateral_second_distance": round(float(second_distance), 6),
            "lateral_prototype_margin": round(float(prototype_margin), 6),
            "lateral_fallback_used": final_label == LATERAL_BOUNDARY_LABEL,
            "lateral_out_of_distribution": "out_of_distribution" in reasons,
            "lateral_classifier_label": classifier_label,
        }

    def _scale(self, vector: np.ndarray) -> np.ndarray:
        scale = np.where(np.abs(self.feature_scale) < 1e-9, 1.0, self.feature_scale)
        return (np.asarray(vector, dtype=float) - self.feature_mean) / scale

    @staticmethod
    def _second_other_label(ordered: list[tuple[str, str, float]], best_label: str) -> tuple[str, str, float]:
        for label, source, distance in ordered[1:]:
            if label != best_label:
                return label, source, distance
        label, source, distance = ordered[0]
        return label, source, distance + 1.0

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
            "labels": [STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL],
            "fallback_label": LATERAL_BOUNDARY_LABEL,
            "feature_names": self.feature_names or LATERAL_FEATURE_NAMES,
            "margin_threshold": self.margin_threshold,
            "confidence_threshold": self.confidence_threshold,
            "distance_thresholds": self.distance_thresholds or {},
            "pair_margin_thresholds": self.pair_margin_thresholds or {},
            "prototype_sources": self.prototype_sources,
        }


class TwoStageLateralRecognizer:
    def __init__(
        self,
        parent_recognizer: object,
        lateral_model: LateralFineModel,
        model_version: str = "v2_3_candidate",
        parent_model_version: str = "v2_2_candidate",
    ) -> None:
        self.parent_recognizer = parent_recognizer
        self.lateral_model = lateral_model
        self.model_version = model_version
        self.parent_model_version = parent_model_version
        self.lateral_submodel_version = lateral_model.submodel_version

    def predict_posture(self, window: np.ndarray) -> dict[str, Any]:
        parent = dict(self.parent_recognizer.predict_posture(window))
        feature_vector, feature_map = extract_lateral_features(window)
        should_run, gate_reasons = should_run_lateral_subclassifier(parent, feature_map)
        payload = dict(parent)
        payload.update(
            {
                "model_version": self.model_version,
                "parent_model_version": self.parent_model_version,
                "lateral_submodel_version": self.lateral_submodel_version,
                "lateral_subclassifier_triggered": should_run,
                "lateral_gate_reason": "; ".join(gate_reasons) if gate_reasons else "",
                "lateral_posture_label": None,
                "lateral_confidence": None,
                "lateral_margin": None,
                "lateral_boundary": False,
                "lateral_boundary_reasons": [],
                "lateral_prototype_label": None,
                "lateral_prototype_distance": None,
                "lateral_fallback_used": False,
            }
        )
        if not should_run:
            return payload
        lateral = self.lateral_model.predict_from_features(feature_vector)
        lateral["lateral_feature_summary"] = feature_map
        payload.update(lateral)
        payload["label"] = lateral["final_display_label"]
        # Keep the top-level stream displayable through the existing smoother.
        # Detailed lateral confidence/margin remain in dedicated diagnostic fields.
        payload["confidence"] = parent.get("confidence", lateral.get("lateral_confidence"))
        payload["second_label"] = lateral.get("lateral_second_label") or parent.get("second_label")
        payload["margin"] = parent.get("margin", lateral.get("lateral_margin"))
        payload["is_boundary"] = False
        payload["boundary_reasons"] = list(parent.get("boundary_reasons") or [])
        return payload


def save_lateral_fine_model(path: Path | str, model: LateralFineModel) -> None:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_lateral_fine_model(path: Path | str) -> LateralFineModel:
    import joblib

    return joblib.load(path)


def save_lateral_prototype_bank(path: Path | str, model: LateralFineModel) -> None:
    payload = {
        "labels": [STANDARD_SIDE_SITTING_LABEL, DIAGONAL_SITTING_LABEL, SIDE_LEANING_LABEL],
        "fallback_label": LATERAL_BOUNDARY_LABEL,
        "feature_names": model.feature_names or LATERAL_FEATURE_NAMES,
        "prototypes": {label: [proto.tolist() for proto in protos] for label, protos in model.prototypes.items()},
        "prototype_sources": model.prototype_sources,
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "distance_thresholds": model.distance_thresholds or {},
        "pair_margin_thresholds": model.pair_margin_thresholds or {},
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
