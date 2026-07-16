from __future__ import annotations

from collections import Counter, deque


UNCERTAIN_LABEL = "边界/不确定"


class PredictionSmoother:
    def __init__(
        self,
        vote_window: int = 7,
        switch_confirmations: int = 3,
        min_confidence: float = 0.55,
        min_margin: float = 0.10,
    ) -> None:
        self.vote_window = max(1, int(vote_window))
        self.switch_confirmations = max(1, int(switch_confirmations))
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self._recent: deque[str] = deque(maxlen=self.vote_window)
        self._current_label: str | None = None
        self._pending_label: str | None = None
        self._pending_count = 0

    def reset(self) -> None:
        self._recent.clear()
        self._current_label = None
        self._pending_label = None
        self._pending_count = 0

    def update(self, prediction: dict[str, object]) -> dict[str, object]:
        low_confidence = float(prediction.get("confidence", 0.0)) < self.min_confidence
        low_margin = float(prediction.get("margin", 0.0)) < self.min_margin
        raw_boundary = bool(prediction.get("is_boundary", False))
        if low_confidence or low_margin or raw_boundary:
            result = dict(prediction)
            reasons = list(prediction.get("boundary_reasons") or [])
            has_model_reasons = bool(reasons)
            if low_confidence and not has_model_reasons:
                reasons.append(f"confidence<{self.min_confidence:.2f}")
            if low_margin and not has_model_reasons:
                reasons.append(f"margin<{self.min_margin:.2f}")
            if raw_boundary and not reasons:
                reasons.append("raw_boundary")
            result["label"] = UNCERTAIN_LABEL
            result["is_boundary"] = True
            result["raw_label"] = prediction.get("label")
            result["raw_confidence"] = prediction.get("confidence")
            result["boundary_reason"] = "; ".join(dict.fromkeys(str(item) for item in reasons))
            return result

        raw_label = str(prediction["label"])
        self._recent.append(raw_label)
        voted_label = Counter(self._recent).most_common(1)[0][0]
        if self._current_label is None:
            self._current_label = voted_label
        elif raw_label != self._current_label:
            if raw_label == self._pending_label:
                self._pending_count += 1
            else:
                self._pending_label = raw_label
                self._pending_count = 1
            if self._pending_count >= self.switch_confirmations:
                self._current_label = raw_label
                self._pending_label = None
                self._pending_count = 0
        else:
            self._pending_label = None
            self._pending_count = 0

        result = dict(prediction)
        result["raw_label"] = raw_label
        result["raw_confidence"] = prediction.get("confidence")
        result["boundary_reason"] = None
        result["label"] = self._current_label
        result["is_boundary"] = False
        return result
