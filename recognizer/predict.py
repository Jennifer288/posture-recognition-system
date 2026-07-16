from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .data_loader import read_sensor_csv, stable_frames
from .feature_extractor import windowed_frames
from .prototype_bank import PrototypeBank
from .recognizer import PosturePrediction, PrototypeRecognizer


def prediction_to_dict(prediction: PosturePrediction) -> dict[str, object]:
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "second_label": prediction.second_label,
        "margin": prediction.margin,
        "is_boundary": prediction.is_boundary,
        "best_distance": prediction.best_distance,
        "matched_prototype_id": prediction.matched_prototype_id,
    }


def predict_csv(
    csv_path: Path | str,
    bank_path: Path | str,
    window: int = 8,
    step: int = 2,
) -> dict[str, object]:
    bank = PrototypeBank.load(bank_path)
    recognizer = PrototypeRecognizer(bank)
    _, frames = read_sensor_csv(csv_path)
    stable = stable_frames(frames)
    windows = windowed_frames(stable, window=window, step=step)
    predictions = [prediction_to_dict(recognizer.predict_posture(sample)) for sample in windows]
    labels = [str(item["label"]) for item in predictions]
    majority = Counter(labels).most_common(1)[0][0] if labels else None
    boundary_rate = sum(1 for item in predictions if item["is_boundary"]) / max(len(predictions), 1)
    mean_confidence = sum(float(item["confidence"]) for item in predictions) / max(len(predictions), 1)
    return {
        "csv_file": Path(csv_path).name,
        "window_count": len(predictions),
        "label": majority,
        "confidence": round(mean_confidence, 4),
        "boundary_rate": round(boundary_rate, 4),
        "windows": predictions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Recognizer V1 on one CSV file using a saved prototype bank.")
    parser.add_argument("csv", help="Sensor CSV to replay.")
    parser.add_argument("--model", required=True, help="Path to prototype_bank_v1.json.")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--step", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(predict_csv(args.csv, args.model, window=args.window, step=args.step), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
