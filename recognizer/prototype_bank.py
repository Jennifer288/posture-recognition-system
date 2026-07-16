from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .feature_extractor import FEATURE_DIM


@dataclass(frozen=True)
class Prototype:
    prototype_id: str
    label: str
    vector: np.ndarray
    mirror_aware: bool = False
    source_files: tuple[str, ...] = field(default_factory=tuple)
    source_group: str = ""

    def __post_init__(self) -> None:
        vec = np.asarray(self.vector, dtype=float)
        if vec.shape != (FEATURE_DIM,):
            raise ValueError(f"Prototype vector must have shape ({FEATURE_DIM},), got {vec.shape}")
        object.__setattr__(self, "vector", vec)
        object.__setattr__(self, "source_files", tuple(self.source_files))

    def to_dict(self) -> dict[str, object]:
        return {
            "prototype_id": self.prototype_id,
            "label": self.label,
            "vector": self.vector.tolist(),
            "mirror_aware": self.mirror_aware,
            "source_files": list(self.source_files),
            "source_group": self.source_group,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Prototype":
        return cls(
            prototype_id=str(data["prototype_id"]),
            label=str(data["label"]),
            vector=np.asarray(data["vector"], dtype=float),
            mirror_aware=bool(data.get("mirror_aware", False)),
            source_files=tuple(str(item) for item in data.get("source_files", [])),
            source_group=str(data.get("source_group", "")),
        )


class PrototypeBank:
    def __init__(
        self,
        prototypes: Iterable[Prototype],
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        class_thresholds: dict[str, float] | None = None,
        margin_threshold: float = 0.0,
        label_taxonomy: dict[str, str] | None = None,
    ) -> None:
        self.prototypes = list(prototypes)
        if not self.prototypes:
            raise ValueError("PrototypeBank requires at least one prototype")
        self.feature_mean = np.asarray(feature_mean, dtype=float) if feature_mean is not None else np.zeros(FEATURE_DIM)
        self.feature_std = np.asarray(feature_std, dtype=float) if feature_std is not None else np.ones(FEATURE_DIM)
        self.feature_std[self.feature_std == 0] = 1.0
        if self.feature_mean.shape != (FEATURE_DIM,) or self.feature_std.shape != (FEATURE_DIM,):
            raise ValueError("feature_mean and feature_std must match feature dimension")
        self.class_thresholds = dict(class_thresholds or {})
        self.margin_threshold = float(margin_threshold)
        self.label_taxonomy = dict(label_taxonomy or {})

    @property
    def labels(self) -> list[str]:
        return sorted({prototype.label for prototype in self.prototypes})

    def standardized(self, feature: np.ndarray) -> np.ndarray:
        vec = np.asarray(feature, dtype=float)
        if vec.shape != (FEATURE_DIM,):
            raise ValueError(f"Feature vector must have shape ({FEATURE_DIM},), got {vec.shape}")
        return (vec - self.feature_mean) / self.feature_std

    def prototypes_for_label(self, label: str) -> list[Prototype]:
        return [prototype for prototype in self.prototypes if prototype.label == label]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": "recognizer_v1_prototype_bank",
            "feature_dim": FEATURE_DIM,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "class_thresholds": self.class_thresholds,
            "margin_threshold": self.margin_threshold,
            "label_taxonomy": self.label_taxonomy,
            "prototypes": [prototype.to_dict() for prototype in self.prototypes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PrototypeBank":
        return cls(
            prototypes=[Prototype.from_dict(item) for item in data["prototypes"]],
            feature_mean=np.asarray(data.get("feature_mean", np.zeros(FEATURE_DIM)), dtype=float),
            feature_std=np.asarray(data.get("feature_std", np.ones(FEATURE_DIM)), dtype=float),
            class_thresholds={str(k): float(v) for k, v in data.get("class_thresholds", {}).items()},
            margin_threshold=float(data.get("margin_threshold", 0.0)),
            label_taxonomy={str(k): str(v) for k, v in data.get("label_taxonomy", {}).items()},
        )

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "PrototypeBank":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
