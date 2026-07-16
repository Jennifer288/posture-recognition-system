from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .frame_reader import CSVReplayReader
from .rf_recognizer import load_hybrid_recognizer
from .seat_analyzer import SeatAnalyzer


@dataclass(frozen=True)
class CsvStreamSummary:
    csv_file: str
    frame_count: int
    fps: float
    first_recognition_delay_s: float | None
    recognizing_frames: int
    label_switch_count: int
    low_confidence_rate: float
    boundary_rate: float
    most_common_label: str | None
    status_counts: dict[str, int]


def run_csv_stream(
    csv_path: Path | str,
    model_path: Path | str,
    prototype_bank_path: Path | str | None = None,
    fps: float = 20.0,
    window_seconds: float = 1.5,
    settle_seconds: float = 1.0,
    vote_window: int = 7,
    switch_confirmations: int = 3,
    emit_events: bool = False,
) -> tuple[CsvStreamSummary, list[dict[str, object]]]:
    reader = CSVReplayReader(csv_path)
    recognizer = load_hybrid_recognizer(model_path, prototype_bank_path)
    analyzer = SeatAnalyzer(
        recognizer=recognizer,
        fps=fps,
        window_seconds=window_seconds,
        settle_seconds=settle_seconds,
    )
    events = []
    first_seated_frame: int | None = None
    first_recognized_frame: int | None = None
    last_label: str | None = None
    switches = 0
    recognizing = 0
    low_conf = 0
    boundary = 0
    labels = []
    status_counts: Counter[str] = Counter()

    index = 0
    while True:
        try:
            frame = reader.read_frame()
        except EOFError:
            break
        payload = analyzer.update(frame)
        status = "POSTURE" if payload.get("posture") else str(payload["seat_state"])
        status_counts[status] += 1
        if payload["occupancy_state"] == "HUMAN" and first_seated_frame is None:
            first_seated_frame = index
        if payload.get("posture") is not None or payload.get("seat_state") == "HUMAN_RECOGNIZING":
            recognizing += 1
            if first_recognized_frame is None:
                first_recognized_frame = index
            label = str(payload.get("posture") or "边界/不确定")
            labels.append(label)
            low_conf += int(float(payload.get("posture_confidence") or 0.0) < 0.55)
            boundary += int(bool(payload.get("is_boundary", False)))
            if last_label is not None and label != last_label:
                switches += 1
            last_label = label
        if payload["occupancy_state"] != "HUMAN":
            last_label = None

        event = {
            "frame": index,
            "time_s": round(index / fps, 3),
            "status": status,
            "status_note": _status_note(status),
            "occupancy_state": payload["occupancy_state"],
            "seat_state": payload["seat_state"],
            "total_pressure": payload["occupancy_features"]["total_pressure"],
            "label": payload.get("posture"),
            "confidence": payload.get("posture_confidence"),
            "second_label": payload.get("second_label") if payload else None,
            "boundary": payload.get("is_boundary") if payload else None,
            "reason": payload.get("reason"),
        }
        if emit_events:
            print(json.dumps(event, ensure_ascii=False))
        events.append(event)
        index += 1

    delay = None
    if first_seated_frame is not None and first_recognized_frame is not None:
        delay = (first_recognized_frame - first_seated_frame) / fps
    most_common = Counter(labels).most_common(1)[0][0] if labels else None
    summary = CsvStreamSummary(
        csv_file=Path(csv_path).name,
        frame_count=index,
        fps=fps,
        first_recognition_delay_s=delay,
        recognizing_frames=recognizing,
        label_switch_count=switches,
        low_confidence_rate=low_conf / max(recognizing, 1),
        boundary_rate=boundary / max(recognizing, 1),
        most_common_label=most_common,
        status_counts=dict(sorted(status_counts.items())),
    )
    return summary, events


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay one pressure CSV through the RF V1 realtime chain.")
    parser.add_argument("csv", help="Sensor CSV to replay.")
    parser.add_argument("--model", default="recognizer/models/rf_posture_v1.joblib")
    parser.add_argument("--prototype-bank", default="recognizer/models/prototype_bank_v1.json")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--window-seconds", type=float, default=1.5)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--emit-events", action="store_true")
    parser.add_argument("--summary-output", default="recognizer/outputs/realtime_csv_simulation_summary.json")
    args = parser.parse_args(argv)

    summary, _ = run_csv_stream(
        args.csv,
        args.model,
        prototype_bank_path=args.prototype_bank,
        fps=args.fps,
        window_seconds=args.window_seconds,
        settle_seconds=args.settle_seconds,
        emit_events=args.emit_events,
    )
    payload = summary.__dict__
    target = Path(args.summary_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _status_note(status: str) -> str:
    if status == "LOAD_BELOW_THRESHOLD":
        return "微弱可检测负载，低于稳定占用阈值，不进入姿势识别"
    if status == "EMPTY":
        return "无可检测负载"
    if status == "OBJECT":
        return "可检测非人体占用，不进入姿势识别"
    if status == "UNKNOWN":
        return "占用证据不足或冲突，不进入姿势识别"
    if status == "HUMAN_STABILIZING":
        return "检测到人体，等待稳定窗口"
    if status == "POSTURE":
        return "人体稳定，输出姿势"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
