from __future__ import annotations

import csv
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from recognizer.offline_posture_analyzer import (
    DEFAULT_MODEL_VERSION,
    OfflineFramePrediction,
    OfflinePostureAnalyzer,
    detect_initial_empty_frames,
    export_offline_analysis,
    merge_posture_segments,
    resolve_fps,
    resolve_orientation,
    summarize_analysis,
)
from recognizer.offline_serial_parser import OfflineParseStats, OfflineSerialParseResult, SerialInputSelection
from recognizer.serial_gui_core import apply_orientation


def make_parse_result(tmp: str, frames: list[np.ndarray], metadata: dict[str, object] | None = None) -> OfflineSerialParseResult:
    path = Path(tmp) / "raw_stream.bin"
    path.write_bytes(b"serial-bytes")
    selection = SerialInputSelection(
        original_path=path,
        input_path=path,
        input_type="BIN",
        metadata_path=None,
        metadata=dict(metadata or {}),
        warnings=[],
    )
    stats = OfflineParseStats(
        input_type="BIN",
        total_bytes=12,
        valid_packets=len(frames),
        invalid_packets=0,
        discarded_bytes=0,
        trailing_incomplete_bytes=0,
        total_frames=len(frames),
        parser_warnings=[],
    )
    return OfflineSerialParseResult(
        selection=selection,
        frames=[np.asarray(frame, dtype=np.float32) for frame in frames],
        raw_packets=[b"packet"] * len(frames),
        checksums=[0] * len(frames),
        stats=stats,
    )


def frame(value: float) -> np.ndarray:
    return np.full((16, 16), value, dtype=np.float32)


def prediction(
    index: int,
    *,
    occupancy: str = "HUMAN",
    posture: str | None = "端正坐姿",
    confidence: float | None = 0.9,
    boundary: bool = False,
) -> OfflineFramePrediction:
    return OfflineFramePrediction(
        frame_index=index,
        time_s=index / 10.0,
        occupancy_state=occupancy,
        seat_state=occupancy,
        posture=posture,
        posture_confidence=confidence,
        raw_label=posture,
        raw_confidence=confidence,
        second_label="前倾端坐",
        margin=0.4,
        is_boundary=boundary,
        boundary_reason="low margin" if boundary else None,
        prototype_diagnosis="proto=端正坐姿",
        parent_posture_label=None,
        fine_posture_label=None,
        lateral_posture_label=None,
        final_display_label=posture,
        selected_branch=None,
        inference_ms=1.0,
        total_pressure=100.0,
        max_pressure=10.0,
        active_points=20,
        display_status="POSTURE" if occupancy == "HUMAN" else occupancy,
    )


class FakeRecognizer:
    def __init__(self, outputs: list[dict[str, object]] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.predict_frames: list[np.ndarray] = []
        self.reset_count = 0
        self.calibration_frame_count = 0
        self.model_version = DEFAULT_MODEL_VERSION

    def reset(self) -> None:
        self.reset_count += 1

    def calibrate(self, frames: np.ndarray) -> dict[str, object]:
        self.calibration_frame_count = len(frames)
        return {"calibrated": True, "frames": len(frames)}

    def predict(self, item: np.ndarray) -> dict[str, object]:
        self.predict_frames.append(np.array(item, copy=True))
        if self.outputs:
            return self.outputs.pop(0)
        total = float(np.asarray(item).sum())
        if total <= 80.0:
            return {"occupancy": "EMPTY", "seat_state": "EMPTY", "posture": None}
        return {
            "occupancy": "HUMAN",
            "seat_state": "HUMAN_RECOGNIZING",
            "posture": "端正坐姿",
            "posture_confidence": 0.9,
            "raw_label": "端正坐姿",
            "raw_confidence": 0.9,
            "second_label": "前倾端坐",
            "margin": 0.5,
            "is_boundary": False,
            "occupancy_features": {"total_pressure": total, "active_points": int((item > 15).sum())},
        }

    def artifact_identity(self) -> dict[str, object]:
        return {"model_version": self.model_version, "model_path": "fake.joblib"}


class OfflinePostureAnalyzerTest(unittest.TestCase):
    def test_resolve_orientation_uses_metadata_and_warns_when_missing(self) -> None:
        self.assertEqual(resolve_orientation({"orientation": "左右翻转"}, fallback_orientation="原始")[0], "左右翻转")
        orientation, warnings = resolve_orientation({}, fallback_orientation="上下翻转")
        self.assertEqual(orientation, "上下翻转")
        self.assertTrue(warnings)

    def test_resolve_fps_uses_metadata_inferred_or_default(self) -> None:
        self.assertEqual(resolve_fps({"fps": 25})[:2], (25.0, "metadata"))
        self.assertEqual(resolve_fps({"valid_frames_saved": 60, "duration_s": 3})[:2], (20.0, "metadata_inferred"))
        fps, source, warnings = resolve_fps({})
        self.assertEqual((fps, source), (20.0, "default"))
        self.assertTrue(warnings)

    def test_initial_empty_detection_requires_reliable_empty_run(self) -> None:
        info, empty = detect_initial_empty_frames([frame(0)] * 10 + [frame(30)], fps=10)
        self.assertEqual(info.calibration_status, "CALIBRATED_FROM_INITIAL_EMPTY")
        self.assertEqual(len(empty), 10)

        info, empty = detect_initial_empty_frames([frame(30), frame(0), frame(0)], fps=10)
        self.assertEqual(info.calibration_status, "NO_RELIABLE_EMPTY_BASELINE")
        self.assertEqual(empty, [])

    def test_analyze_applies_metadata_orientation_once_and_resets_recognizer(self) -> None:
        source = np.arange(256, dtype=np.float32).reshape(16, 16)
        fake = FakeRecognizer()
        with TemporaryDirectory() as tmp:
            parse_result = make_parse_result(tmp, [source], metadata={"orientation": "左右翻转", "fps": 20})
            analyzer = OfflinePostureAnalyzer(recognizer_factory=lambda **_kwargs: fake)

            result = analyzer.analyze(parse_result, orientation="上下翻转")

        self.assertEqual(result.orientation, "左右翻转")
        self.assertEqual(fake.reset_count, 1)
        np.testing.assert_array_equal(fake.predict_frames[0], apply_orientation(source, "左右翻转"))

    def test_metadata_label_does_not_change_prediction(self) -> None:
        outputs = [
            {
                "occupancy": "HUMAN",
                "seat_state": "HUMAN_RECOGNIZING",
                "posture": "端正坐姿",
                "posture_confidence": 0.91,
                "raw_label": "端正坐姿",
                "raw_confidence": 0.91,
                "margin": 0.4,
                "is_boundary": False,
            }
        ] * 40

        def run(label: str) -> str | None:
            with TemporaryDirectory() as tmp:
                parse_result = make_parse_result(tmp, [frame(20)] * 40, metadata={"label": label, "fps": 20})
                fake = FakeRecognizer(outputs=list(outputs))
                analyzer = OfflinePostureAnalyzer(recognizer_factory=lambda **_kwargs: fake)
                return analyzer.analyze(parse_result).summary.dominant_posture

        self.assertEqual(run("端正坐姿"), "端正坐姿")
        self.assertEqual(run("斜跨坐"), "端正坐姿")

    def test_calibrates_only_when_initial_empty_is_reliable(self) -> None:
        fake = FakeRecognizer()
        with TemporaryDirectory() as tmp:
            parse_result = make_parse_result(tmp, [frame(0)] * 20 + [frame(20)] * 20, metadata={"fps": 20})
            analyzer = OfflinePostureAnalyzer(recognizer_factory=lambda **_kwargs: fake)

            result = analyzer.analyze(parse_result)

        self.assertEqual(result.calibration_info.calibration_status, "CALIBRATED_FROM_INITIAL_EMPTY")
        self.assertEqual(fake.calibration_frame_count, 20)

    def test_merge_segments_and_summary_exclude_empty_stabilizing_and_boundary(self) -> None:
        records = [
            prediction(0, occupancy="EMPTY", posture=None, confidence=None),
            prediction(1, occupancy="HUMAN", posture=None, confidence=None),
            prediction(2, posture="端正坐姿", confidence=0.9),
            prediction(3, posture="端正坐姿", confidence=0.8),
            prediction(4, posture=None, confidence=None, boundary=True),
            prediction(5, posture="前倾端坐", confidence=0.7),
        ]
        segments = merge_posture_segments(records, fps=10)
        summary = summarize_analysis(
            records,
            segments,
            fps=10,
            parser_stats={"valid_packets": len(records)},
            warnings=[],
        )

        self.assertEqual([segment.segment_type for segment in segments], ["EMPTY", "HUMAN_STABILIZING", "POSTURE", "BOUNDARY", "POSTURE"])
        self.assertEqual(summary.posture_frame_counts, {"端正坐姿": 2, "前倾端坐": 1})
        self.assertEqual(summary.boundary_frames, 1)
        self.assertEqual(summary.stable_posture_frames, 3)

    def test_status_thresholds_cover_major_quality_outcomes(self) -> None:
        no_human = [prediction(i, occupancy="EMPTY", posture=None, confidence=None) for i in range(25)]
        self.assertEqual(summarize_analysis(no_human, merge_posture_segments(no_human, fps=10), fps=10, parser_stats={"valid_packets": 25}).result_status, "NO_HUMAN")

        low_conf = [prediction(i, confidence=0.4) for i in range(25)]
        self.assertEqual(summarize_analysis(low_conf, merge_posture_segments(low_conf, fps=10), fps=10, parser_stats={"valid_packets": 25}).result_status, "LOW_CONFIDENCE")

        high_boundary = [prediction(i, boundary=i < 25, posture=None if i < 25 else "端正坐姿") for i in range(50)]
        self.assertEqual(summarize_analysis(high_boundary, merge_posture_segments(high_boundary, fps=10), fps=10, parser_stats={"valid_packets": 50}).result_status, "HIGH_BOUNDARY_RATE")

        success = [prediction(i, posture="端正坐姿") for i in range(30)]
        self.assertEqual(summarize_analysis(success, merge_posture_segments(success, fps=10), fps=10, parser_stats={"valid_packets": 30}).result_status, "SUCCESS")

    def test_status_thresholds_cover_invalid_insufficient_and_mixed_postures(self) -> None:
        empty_parse = summarize_analysis([], [], fps=10, parser_stats={"valid_packets": 0})
        self.assertEqual(empty_parse.result_status, "INVALID_SERIAL_DATA")

        short = [prediction(i, posture="端正坐姿") for i in range(10)]
        self.assertEqual(summarize_analysis(short, merge_posture_segments(short, fps=10), fps=10, parser_stats={"valid_packets": 10}).result_status, "INSUFFICIENT_DATA")

        mixed = [prediction(i, posture="端正坐姿") for i in range(20)] + [prediction(i + 20, posture="前倾端坐") for i in range(20)]
        self.assertEqual(summarize_analysis(mixed, merge_posture_segments(mixed, fps=10), fps=10, parser_stats={"valid_packets": 40}).result_status, "MIXED_POSTURES")

    def test_manual_label_is_only_used_for_consistency_comparison(self) -> None:
        records = [prediction(i, posture="端正坐姿") for i in range(30)]
        summary = summarize_analysis(
            records,
            merge_posture_segments(records, fps=10),
            fps=10,
            parser_stats={"valid_packets": 30},
            manual_label="端正坐姿",
        )
        self.assertEqual(summary.overall_posture, "端正坐姿")
        self.assertTrue(summary.label_matches_overall)

        mismatch = summarize_analysis(
            records,
            merge_posture_segments(records, fps=10),
            fps=10,
            parser_stats={"valid_packets": 30},
            manual_label="斜跨坐",
        )
        self.assertEqual(mismatch.overall_posture, "端正坐姿")
        self.assertFalse(mismatch.label_matches_overall)

    def test_export_writes_report_frames_segments_and_parse_report_without_touching_input(self) -> None:
        with TemporaryDirectory() as tmp:
            parse_result = make_parse_result(tmp, [frame(20)] * 25, metadata={"fps": 10, "label": "端正坐姿"})
            input_before = parse_result.selection.input_path.read_bytes()
            fake = FakeRecognizer()
            result = OfflinePostureAnalyzer(recognizer_factory=lambda **_kwargs: fake).analyze(parse_result)
            outputs = export_offline_analysis(result, Path(tmp) / "exports")

            self.assertEqual(parse_result.selection.input_path.read_bytes(), input_before)
            self.assertTrue(outputs["report"].exists())
            self.assertTrue(outputs["frame_predictions"].exists())
            self.assertTrue(outputs["segments"].exists())
            self.assertTrue(outputs["parse_report"].exists())
            report = json.loads(outputs["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["model_version"], DEFAULT_MODEL_VERSION)
            self.assertEqual(report["input_sha256"], result.input_sha256)
            with outputs["frame_predictions"].open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 25)

    def test_cancel_event_stops_analysis(self) -> None:
        cancel = threading.Event()

        def progress(current: int, _total: int) -> None:
            if current == 3:
                cancel.set()

        with TemporaryDirectory() as tmp:
            parse_result = make_parse_result(tmp, [frame(20)] * 20, metadata={"fps": 20})
            result = OfflinePostureAnalyzer(recognizer_factory=lambda **_kwargs: FakeRecognizer()).analyze(
                parse_result,
                progress_callback=progress,
                cancel_event=cancel,
            )

        self.assertEqual(result.summary.result_status, "ANALYSIS_CANCELLED")
        self.assertEqual(len(result.frame_predictions), 3)


if __name__ == "__main__":
    unittest.main()
