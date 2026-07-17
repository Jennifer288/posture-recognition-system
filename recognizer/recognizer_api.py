from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .feature_extractor import as_frame, as_frames
from .leanback_subclassifier import TwoStageLeanbackRecognizer, load_leanback_fine_model
from .lateral_subclassifier import TwoStageLateralRecognizer, load_lateral_fine_model
from .lateral_merged_subclassifier import TwoStageLateralMergedRecognizer, load_lateral_merged_fine_model
from .lateral_merged_subclassifier_v241 import (
    TwoStageLateralMergedRecognizerV241,
    load_lateral_merged_fine_model_v241,
)
from .lateral_merged_subclassifier_v242 import (
    TwoStageLateralMergedRecognizerV242,
    load_lateral_merged_fine_model_v242,
)
from .occupancy_detector import OccupancyDetector
from .rf_recognizer import load_hybrid_recognizer
from .seat_analyzer import SeatAnalyzer


PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR / "models"
DEFAULT_MODEL_CONFIG_PATH = MODELS_DIR / "default_model.json"
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / "rf_posture_v1.joblib"
DEFAULT_PROTOTYPE_BANK_PATH = PACKAGE_DIR / "models" / "prototype_bank_v1.json"
DEFAULT_METADATA_PATH = PACKAGE_DIR / "models" / "rf_posture_v1.metadata.json"
MODEL_VERSION_ARTIFACTS = {
    "v1": {
        "model": DEFAULT_MODEL_PATH,
        "prototype_bank": DEFAULT_PROTOTYPE_BANK_PATH,
        "metadata": DEFAULT_METADATA_PATH,
        "runtime_config": None,
    },
    "v2_candidate": {
        "model": PACKAGE_DIR / "models" / "rf_posture_v2_candidate.joblib",
        "prototype_bank": PACKAGE_DIR / "models" / "prototype_bank_v2_candidate.json",
        "metadata": PACKAGE_DIR / "models" / "rf_posture_v2_candidate.metadata.json",
        "runtime_config": PACKAGE_DIR / "models" / "rf_posture_v2_candidate.runtime_config.json",
    },
    "v2_1_candidate": {
        "model": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.joblib",
        "prototype_bank": PACKAGE_DIR / "models" / "prototype_bank_v2_1_candidate.json",
        "metadata": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.metadata.json",
        "runtime_config": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.runtime_config.json",
    },
    "v2_2_candidate": {
        "model": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.joblib",
        "prototype_bank": PACKAGE_DIR / "models" / "prototype_bank_v2_1_candidate.json",
        "metadata": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.metadata.json",
        "runtime_config": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.runtime_config.json",
        "submodel": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.joblib",
        "subprototype_bank": PACKAGE_DIR / "models" / "leanback_prototype_bank_v2_2_candidate.json",
        "subruntime_config": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.runtime_config.json",
        "bundle": PACKAGE_DIR / "models" / "v2_2_candidate.model_bundle.json",
    },
    "v2_3_candidate": {
        "model": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.joblib",
        "prototype_bank": PACKAGE_DIR / "models" / "prototype_bank_v2_1_candidate.json",
        "metadata": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.metadata.json",
        "runtime_config": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.runtime_config.json",
        "submodel": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.joblib",
        "subprototype_bank": PACKAGE_DIR / "models" / "leanback_prototype_bank_v2_2_candidate.json",
        "subruntime_config": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.runtime_config.json",
        "lateral_submodel": PACKAGE_DIR / "models" / "lateral_subclassifier_v2_3_candidate.joblib",
        "lateral_prototype_bank": PACKAGE_DIR / "models" / "lateral_prototype_bank_v2_3_candidate.json",
        "lateral_runtime_config": PACKAGE_DIR / "models" / "lateral_subclassifier_v2_3_candidate.runtime_config.json",
        "bundle": PACKAGE_DIR / "models" / "v2_3_candidate.model_bundle.json",
    },
    "v2_3_1_candidate": {
        "model": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.joblib",
        "prototype_bank": PACKAGE_DIR / "models" / "prototype_bank_v2_1_candidate.json",
        "metadata": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.metadata.json",
        "runtime_config": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.runtime_config.json",
        "submodel": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.joblib",
        "subprototype_bank": PACKAGE_DIR / "models" / "leanback_prototype_bank_v2_2_candidate.json",
        "subruntime_config": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.runtime_config.json",
        "lateral_submodel": PACKAGE_DIR / "models" / "lateral_subclassifier_v2_3_1_candidate.joblib",
        "lateral_prototype_bank": PACKAGE_DIR / "models" / "lateral_prototype_bank_v2_3_1_candidate.json",
        "lateral_runtime_config": PACKAGE_DIR / "models" / "lateral_subclassifier_v2_3_1_candidate.runtime_config.json",
        "bundle": PACKAGE_DIR / "models" / "v2_3_1_candidate.model_bundle.json",
    },
    "v2_4_candidate": {
        "model": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.joblib",
        "prototype_bank": PACKAGE_DIR / "models" / "prototype_bank_v2_1_candidate.json",
        "metadata": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.metadata.json",
        "runtime_config": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.runtime_config.json",
        "submodel": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.joblib",
        "subprototype_bank": PACKAGE_DIR / "models" / "leanback_prototype_bank_v2_2_candidate.json",
        "subruntime_config": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.runtime_config.json",
        "lateral_submodel": PACKAGE_DIR / "models" / "lateral_merged_subclassifier_v2_4_candidate.joblib",
        "lateral_prototype_bank": PACKAGE_DIR / "models" / "lateral_merged_prototype_bank_v2_4_candidate.json",
        "lateral_runtime_config": PACKAGE_DIR / "models" / "lateral_merged_subclassifier_v2_4_candidate.runtime_config.json",
        "bundle": PACKAGE_DIR / "models" / "v2_4_candidate.model_bundle.json",
    },
    "v2_4_1_candidate": {
        "model": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.joblib",
        "prototype_bank": PACKAGE_DIR / "models" / "prototype_bank_v2_1_candidate.json",
        "metadata": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.metadata.json",
        "runtime_config": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.runtime_config.json",
        "submodel": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.joblib",
        "subprototype_bank": PACKAGE_DIR / "models" / "leanback_prototype_bank_v2_2_candidate.json",
        "subruntime_config": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.runtime_config.json",
        "lateral_submodel": PACKAGE_DIR / "models" / "lateral_merged_subclassifier_v2_4_1_candidate.joblib",
        "lateral_prototype_bank": PACKAGE_DIR / "models" / "lateral_merged_prototype_bank_v2_4_1_candidate.json",
        "lateral_runtime_config": PACKAGE_DIR / "models" / "lateral_merged_subclassifier_v2_4_1_candidate.runtime_config.json",
        "bundle": PACKAGE_DIR / "models" / "v2_4_1_candidate.model_bundle.json",
    },
    "v2_4_2_candidate": {
        "model": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.joblib",
        "prototype_bank": PACKAGE_DIR / "models" / "prototype_bank_v2_1_candidate.json",
        "metadata": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.metadata.json",
        "runtime_config": PACKAGE_DIR / "models" / "rf_posture_v2_1_candidate.runtime_config.json",
        "submodel": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.joblib",
        "subprototype_bank": PACKAGE_DIR / "models" / "leanback_prototype_bank_v2_2_candidate.json",
        "subruntime_config": PACKAGE_DIR / "models" / "leanback_subclassifier_v2_2_candidate.runtime_config.json",
        "lateral_submodel": PACKAGE_DIR / "models" / "lateral_merged_subclassifier_v2_4_2_candidate.joblib",
        "lateral_prototype_bank": PACKAGE_DIR / "models" / "lateral_merged_prototype_bank_v2_4_2_candidate.json",
        "lateral_runtime_config": PACKAGE_DIR / "models" / "lateral_merged_subclassifier_v2_4_2_candidate.runtime_config.json",
        "bundle": PACKAGE_DIR / "models" / "v2_4_2_candidate.model_bundle.json",
    },
}


def default_model_version(config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> str:
    """Return the configured default runtime model version."""

    path = Path(config_path)
    if not path.exists():
        return "v1"
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("model_version")
    if not isinstance(version, str):
        raise ValueError(f"default model config must contain a string model_version: {path}")
    if version not in MODEL_VERSION_ARTIFACTS:
        valid = ", ".join(sorted(MODEL_VERSION_ARTIFACTS))
        raise ValueError(f"Unknown default model_version {version!r}. Expected one of: {valid}")
    return version


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists() or not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Recognizer:
    """Hardware-facing Recognizer V1 facade.

    External callers only need to create this class and pass one 16x16 pressure
    frame at a time to ``predict``. The occupancy gate, stable-window buffering,
    RF model, prototype diagnosis, and smoothing are intentionally hidden here.
    """

    def __init__(
        self,
        model_version: str | None = None,
        model_path: str | Path | None = None,
        prototype_bank_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        runtime_config_path: str | Path | None = None,
        fps: float = 20.0,
        window_seconds: float = 1.5,
        settle_seconds: float = 1.0,
        analyzer: SeatAnalyzer | object | None = None,
        posture_recognizer: object | None = None,
    ) -> None:
        resolved_model_version = default_model_version() if model_version is None else model_version
        artifacts = self._artifacts_for_version(resolved_model_version)
        self.model_version = resolved_model_version
        self.model_path = Path(model_path) if model_path is not None else artifacts["model"]
        self.prototype_bank_path = (
            Path(prototype_bank_path) if prototype_bank_path is not None else artifacts["prototype_bank"]
        )
        self.metadata_path = Path(metadata_path) if metadata_path is not None else artifacts["metadata"]
        default_runtime = artifacts.get("runtime_config")
        self.runtime_config_path = (
            Path(runtime_config_path)
            if runtime_config_path is not None
            else Path(default_runtime)
            if default_runtime is not None
            else None
        )
        self.submodel_path = artifacts.get("submodel")
        self.subprototype_bank_path = artifacts.get("subprototype_bank")
        self.subruntime_config_path = artifacts.get("subruntime_config")
        self.lateral_submodel_path = artifacts.get("lateral_submodel")
        self.lateral_prototype_bank_path = artifacts.get("lateral_prototype_bank")
        self.lateral_runtime_config_path = artifacts.get("lateral_runtime_config")
        self.model_bundle_path = artifacts.get("bundle")
        self.fps = float(fps)
        self.window_seconds = float(window_seconds)
        self.settle_seconds = float(settle_seconds)
        self.metadata = self._load_metadata(self.metadata_path)
        self._injected_analyzer = analyzer is not None
        self._posture_recognizer = posture_recognizer
        if analyzer is not None:
            self._analyzer = analyzer
        else:
            if self._posture_recognizer is None:
                self._posture_recognizer = self._load_posture_recognizer()
            self._analyzer = self._build_analyzer()

    def predict(self, frame: np.ndarray) -> dict[str, object]:
        """Analyze one realtime 16x16 pressure frame.

        Raises:
            ValueError: if ``frame`` is not shaped ``(16, 16)``.
        """

        arr = as_frame(frame)
        raw = self._analyzer.update(arr)
        return self._public_payload(raw)

    def reset(self) -> None:
        """Clear realtime buffers and previous posture state."""

        reset = getattr(self._analyzer, "reset", None)
        if callable(reset):
            reset()

    def calibrate(self, frame: np.ndarray | None = None, frames: np.ndarray | None = None) -> dict[str, object]:
        """Recalibrate the empty-seat baseline without exposing internals.

        ``frame`` calibrates from one empty 16x16 frame. ``frames`` accepts a
        stack shaped ``(n, 16, 16)``. Calling without either argument clears the
        realtime buffers and starts fresh baseline collection on future frames.
        """

        if frame is not None and frames is not None:
            raise ValueError("Pass either frame or frames, not both")

        calibration_frames: np.ndarray | None = None
        if frame is not None:
            calibration_frames = as_frame(frame).reshape(1, 16, 16)
        elif frames is not None:
            calibration_frames = as_frames(frames)

        if self._injected_analyzer:
            if calibration_frames is not None:
                frame_count = int(len(calibration_frames))
            else:
                frame_count = 0
            return {"calibrated": True, "frames": frame_count, "mode": "external_analyzer"}

        detector = OccupancyDetector(fps=self.fps)
        frame_count = 0
        if calibration_frames is not None:
            for item in calibration_frames:
                detector.update(item)
                frame_count += 1
        self._analyzer = self._build_analyzer(occupancy_detector=detector)
        return {"calibrated": True, "frames": frame_count, "mode": "baseline_reset"}

    def _load_posture_recognizer(self) -> object:
        if not self.model_path.exists():
            raise FileNotFoundError(f"RF V1 model not found: {self.model_path}")
        prototype_path = self.prototype_bank_path if self.prototype_bank_path.exists() else None
        parent = load_hybrid_recognizer(self.model_path, prototype_path)
        if self.model_version not in {"v2_2_candidate", "v2_3_candidate", "v2_3_1_candidate", "v2_4_candidate", "v2_4_1_candidate", "v2_4_2_candidate"}:
            return parent
        if self.submodel_path is None or not Path(self.submodel_path).exists():
            raise FileNotFoundError(f"V2.2 leanback submodel not found: {self.submodel_path}")
        fine_model = load_leanback_fine_model(self.submodel_path)
        v22 = TwoStageLeanbackRecognizer(
            parent,
            fine_model,
            model_version="v2_2_candidate",
            parent_model_version="v2_1_candidate",
        )
        if self.model_version == "v2_2_candidate":
            return v22
        if self.lateral_submodel_path is None or not Path(self.lateral_submodel_path).exists():
            raise FileNotFoundError(f"V2.3/V2.4 lateral submodel not found: {self.lateral_submodel_path}")
        if self.model_version == "v2_4_candidate":
            lateral_model = load_lateral_merged_fine_model(self.lateral_submodel_path)
            return TwoStageLateralMergedRecognizer(
                v22,
                lateral_model,
                model_version=self.model_version,
                parent_model_version="v2_2_candidate",
            )
        if self.model_version == "v2_4_1_candidate":
            lateral_model = load_lateral_merged_fine_model_v241(self.lateral_submodel_path)
            return TwoStageLateralMergedRecognizerV241(
                v22,
                lateral_model,
                model_version=self.model_version,
                parent_model_version="v2_2_candidate",
            )
        if self.model_version == "v2_4_2_candidate":
            lateral_model = load_lateral_merged_fine_model_v242(self.lateral_submodel_path)
            return TwoStageLateralMergedRecognizerV242(
                v22,
                lateral_model,
                model_version=self.model_version,
                parent_model_version="v2_2_candidate",
            )
        lateral_model = load_lateral_fine_model(self.lateral_submodel_path)
        return TwoStageLateralRecognizer(
            v22,
            lateral_model,
            model_version=self.model_version,
            parent_model_version="v2_2_candidate",
        )

    def _build_analyzer(self, occupancy_detector: OccupancyDetector | None = None) -> SeatAnalyzer:
        return SeatAnalyzer(
            recognizer=self._posture_recognizer,
            fps=self.fps,
            window_seconds=self.window_seconds,
            settle_seconds=self.settle_seconds,
            occupancy_detector=occupancy_detector,
        )

    @staticmethod
    def _artifacts_for_version(model_version: str) -> dict[str, Path]:
        if model_version not in MODEL_VERSION_ARTIFACTS:
            valid = ", ".join(sorted(MODEL_VERSION_ARTIFACTS))
            raise ValueError(f"Unknown model_version {model_version!r}. Expected one of: {valid}")
        return MODEL_VERSION_ARTIFACTS[model_version]

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_identity(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "prototype_bank_path": str(self.prototype_bank_path),
            "runtime_config_path": None if self.runtime_config_path is None else str(self.runtime_config_path),
            "model_artifact_sha256": sha256_file(self.model_path),
            "metadata_sha256": sha256_file(self.metadata_path),
            "prototype_bank_sha256": sha256_file(self.prototype_bank_path),
            "runtime_config_sha256": sha256_file(self.runtime_config_path),
            "submodel_path": None if self.submodel_path is None else str(self.submodel_path),
            "subprototype_bank_path": None if self.subprototype_bank_path is None else str(self.subprototype_bank_path),
            "subruntime_config_path": None if self.subruntime_config_path is None else str(self.subruntime_config_path),
            "lateral_submodel_path": None if self.lateral_submodel_path is None else str(self.lateral_submodel_path),
            "lateral_prototype_bank_path": None if self.lateral_prototype_bank_path is None else str(self.lateral_prototype_bank_path),
            "lateral_runtime_config_path": None if self.lateral_runtime_config_path is None else str(self.lateral_runtime_config_path),
            "model_bundle_path": None if self.model_bundle_path is None else str(self.model_bundle_path),
            "submodel_sha256": sha256_file(self.submodel_path),
            "subprototype_bank_sha256": sha256_file(self.subprototype_bank_path),
            "subruntime_config_sha256": sha256_file(self.subruntime_config_path),
            "lateral_submodel_sha256": sha256_file(self.lateral_submodel_path),
            "lateral_prototype_bank_sha256": sha256_file(self.lateral_prototype_bank_path),
            "lateral_runtime_config_sha256": sha256_file(self.lateral_runtime_config_path),
            "model_bundle_sha256": sha256_file(self.model_bundle_path),
        }

    @staticmethod
    def _public_payload(raw: dict[str, Any]) -> dict[str, object]:
        occupancy = raw.get("occupancy_state")
        seat_state = raw.get("seat_state")
        posture = raw.get("posture")
        is_human = occupancy == "HUMAN"
        if occupancy != "HUMAN":
            posture = None
        payload = {
            "occupancy": occupancy,
            "occupancy_confidence": raw.get("occupancy_confidence"),
            "seat_state": seat_state,
            "posture": posture,
            "posture_confidence": raw.get("posture_confidence") if is_human else None,
            "second_label": raw.get("second_label") if is_human else None,
            "margin": raw.get("margin") if is_human else None,
            "is_boundary": bool(raw.get("is_boundary", False)),
            "raw_label": raw.get("raw_label") if is_human else None,
            "raw_confidence": raw.get("raw_confidence") if is_human else None,
            "boundary_reason": raw.get("boundary_reason") if is_human else None,
            "prototype_diagnosis": raw.get("prototype_diagnosis") if is_human else None,
            "reason": raw.get("reason"),
            "occupancy_features": raw.get("occupancy_features"),
        }
        for key in [
            "parent_posture_label",
            "fine_posture_label",
            "final_display_label",
            "subclassifier_triggered",
            "subclassifier_gate_reason",
            "fine_confidence",
            "fine_margin",
            "fine_boundary",
            "fine_boundary_reasons",
            "fine_prototype_label",
            "fine_prototype_distance",
            "fallback_used",
            "parent_model_version",
            "submodel_version",
            "lateral_subclassifier_triggered",
            "lateral_gate_reason",
            "lateral_posture_label",
            "lateral_confidence",
            "lateral_margin",
            "lateral_boundary",
            "lateral_boundary_reasons",
            "lateral_prototype_label",
            "lateral_prototype_distance",
            "lateral_fallback_used",
            "lateral_submodel_version",
            "lateral_second_label",
            "lateral_second_distance",
            "lateral_prototype_margin",
            "lateral_out_of_distribution",
            "lateral_temporal_state",
            "lateral_stable_label",
            "lateral_fallback_requested",
            "lateral_merged_label",
            "lateral_prototype_subtype",
            "lateral_second_subtype",
            "parent_raw_lateral_label",
            "label_taxonomy_version",
            "final_priority_branch",
            "selected_branch",
            "override_reason",
            "fallback_reason",
            "lateral_gate_candidate",
            "lateral_distance_z",
            "lateral_classifier_label",
            "lateral_prototype_source",
            "lateral_second_prototype_source",
            "lateral_normalization_applied",
            "lateral_normalization_reason",
            "lateral_normalization_confidence",
            "lateral_physical_evidence_passed",
            "lateral_physical_evidence_reasons",
            "selected_final_branch",
            "final_override_reason",
            "lateral_gate_strong_evidence",
            "lateral_gate_soft_warnings",
            "lateral_gate_hard_reject_reasons",
            "front_back_support_warning",
            "front_back_support_hard_reject",
            "parent_prototype_agreement",
            "lateral_physical_evidence_score",
            "lateral_gate_decision",
            "lateral_gate_decision_reason",
        ]:
            payload[key] = raw.get(key) if is_human else None
        return payload
