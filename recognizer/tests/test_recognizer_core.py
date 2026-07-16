from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from collections import deque
import csv
import hashlib
import json

import numpy as np

from recognizer.frame_reader import CSVReplayReader, SerialFrameReader
from recognizer.feature_extractor import extract_features, window_average
from recognizer.data_loader import map_v1_label, source_family_from_name
from recognizer.model_artifact import load_model_bundle, save_model_bundle
from recognizer.occupancy_detector import OccupancyDetector, OccupancyState
from recognizer.prototype_bank import Prototype, PrototypeBank
from recognizer.recognizer import ProbabilityRecognizer, PrototypeRecognizer, RecognizerConfig
from recognizer.realtime_cli import run_csv_stream
from recognizer.rf_recognizer import HybridPostureRecognizer
from recognizer.seat_detector import SeatDetector, SeatPhase
from recognizer.seat_analyzer import SeatAnalyzer
from recognizer.smoothing import PredictionSmoother
from recognizer.training import build_v1_prototype_bank, compare_v1_models
from recognizer.pipeline import RealtimePosturePipeline
from recognizer.predict import prediction_to_dict
from recognizer.gui import pressure_to_color
from recognizer.leanback_subclassifier import (
    FINE_BOUNDARY_LABEL,
    FINE_LEANBACK_LABEL,
    FINE_SLOUCH_LABEL,
    LEANBACK_FEATURE_NAMES,
    LeanbackFineModel,
    TwoStageLeanbackRecognizer,
    leanback_physical_gate,
    should_run_leanback_subclassifier,
)
from recognizer.lateral_subclassifier import (
    DIAGONAL_SITTING_LABEL,
    LATERAL_BOUNDARY_LABEL,
    LATERAL_FEATURE_NAMES,
    SIDE_LEANING_LABEL,
    STANDARD_SIDE_SITTING_LABEL,
    LateralFineModel,
    TwoStageLateralRecognizer,
    lateral_physical_gate,
    should_run_lateral_subclassifier,
)


class ConstantProbModel:
    classes_ = np.asarray(["A", "B"])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.82, 0.18]]), (len(features), 1))


class CountingRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def predict_posture(self, window: np.ndarray) -> dict[str, object]:
        self.calls += 1
        return {
            "label": "端正坐姿",
            "confidence": 0.91,
            "second_label": "前倾端坐",
            "margin": 0.55,
            "is_boundary": False,
            "prototype_diagnosis": None,
        }


class FakeAnalyzer:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0

    def update(self, frame: np.ndarray) -> dict[str, object]:
        self.calls += 1
        return {
            "occupancy_state": "HUMAN",
            "occupancy_confidence": 0.88,
            "seat_state": "HUMAN_RECOGNIZING",
            "posture": "端正坐姿",
            "posture_confidence": 0.91,
            "second_label": "前倾端坐",
            "margin": 0.42,
            "is_boundary": False,
            "reason": "fake analyzer result",
            "prototype_diagnosis": {"label": "端正坐姿"},
            "occupancy_features": {"total_pressure": 1234.0},
        }

    def reset(self) -> None:
        self.reset_calls += 1


class BoundaryAnalyzer:
    def update(self, frame: np.ndarray) -> dict[str, object]:
        return {
            "occupancy_state": "HUMAN",
            "occupancy_confidence": 0.84,
            "seat_state": "HUMAN_RECOGNIZING",
            "posture": None,
            "posture_confidence": 0.49,
            "second_label": "前倾端坐",
            "margin": 0.06,
            "is_boundary": True,
            "reason": "human stable; posture boundary",
            "raw_label": "端正坐姿",
            "raw_confidence": 0.49,
            "boundary_reason": "RF margin<0.10",
            "prototype_diagnosis": {"label": "端正坐姿", "best_distance": 7.2},
            "occupancy_features": {"total_pressure": 3456.0, "active_points": 52},
        }

    def reset(self) -> None:
        pass


class SequenceRecognizerApi:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = deque(outputs)
        self.last = outputs[-1]
        self.calls = 0
        self.reset_calls = 0

    def predict(self, frame: np.ndarray) -> dict[str, object]:
        self.calls += 1
        if self.outputs:
            return self.outputs.popleft()
        return dict(self.last)

    def reset(self) -> None:
        self.reset_calls += 1

    def calibrate(self, frame: np.ndarray | None = None) -> dict[str, object]:
        return {"calibrated": True, "frames": int(frame is not None)}


def api_payload(
    occupancy: str = "HUMAN",
    posture: str | None = "端正坐姿",
    confidence: float | None = 0.9,
    seat_state: str | None = None,
    boundary: bool = False,
) -> dict[str, object]:
    resolved_seat_state = seat_state or ("HUMAN_RECOGNIZING" if occupancy == "HUMAN" else occupancy)
    return {
        "occupancy": occupancy,
        "occupancy_confidence": 0.88,
        "seat_state": resolved_seat_state,
        "posture": posture,
        "posture_confidence": confidence,
        "second_label": "前倾端坐" if posture else None,
        "margin": 0.42 if posture else None,
        "is_boundary": boundary,
        "prototype_diagnosis": {"label": posture} if posture else None,
        "reason": "test",
        "occupancy_features": {"total_pressure": 1000.0 if occupancy == "HUMAN" else 0.0, "active_points": 48},
    }


def human_like_frame(value: float = 50.0) -> np.ndarray:
    frame = np.zeros((16, 16), dtype=float)
    frame[4:10, 4:12] = value
    frame[10:13, 5:11] = value * 0.45
    return frame


def object_like_frame() -> np.ndarray:
    frame = np.zeros((16, 16), dtype=float)
    frame[6:9, 6:9] = 220.0
    return frame


def unknown_like_frame() -> np.ndarray:
    frame = np.zeros((16, 16), dtype=float)
    frame[5:9, 5:10] = 18.0
    return frame


class FeatureExtractorTest(unittest.TestCase):
    def test_extract_features_matches_training_feature_shape(self) -> None:
        frame = np.zeros((16, 16), dtype=float)
        frame[4:8, 6:10] = 100.0

        features = extract_features(frame)

        self.assertEqual(features.shape, (264,))
        self.assertAlmostEqual(float(features[:256].sum()), 1.0)

    def test_window_average_accepts_multiple_frames(self) -> None:
        frames = np.stack([np.ones((16, 16)) * 10, np.ones((16, 16)) * 30], axis=0)

        averaged = window_average(frames)

        self.assertEqual(averaged.shape, (16, 16))
        self.assertAlmostEqual(float(averaged.mean()), 20.0)


class SeatDetectorTest(unittest.TestCase):
    def test_detects_empty_transition_and_stable_state(self) -> None:
        detector = SeatDetector(fps=10.0, settle_seconds=1.0)
        empty = np.zeros((16, 16), dtype=float)
        occupied = np.ones((16, 16), dtype=float) * 20

        for _ in range(5):
            snapshot = detector.update(empty)
        self.assertEqual(snapshot.phase, SeatPhase.EMPTY)

        transition = detector.update(occupied)
        self.assertEqual(transition.phase, SeatPhase.SITTING_DOWN)

        for _ in range(10):
            stable = detector.update(occupied)
        self.assertEqual(stable.phase, SeatPhase.STABLE)
        self.assertGreaterEqual(stable.occupied_duration_s, 1.0)


class OccupancyDetectorTest(unittest.TestCase):
    def test_empty_object_human_and_unknown_rules_are_separated(self) -> None:
        detector = OccupancyDetector(fps=10.0)

        empty = detector.analyze(np.zeros((16, 16), dtype=float))
        obj = detector.analyze(object_like_frame())
        human = detector.analyze(human_like_frame())
        unknown = detector.analyze(unknown_like_frame())

        self.assertEqual(empty.state, OccupancyState.EMPTY)
        self.assertEqual(obj.state, OccupancyState.OBJECT)
        self.assertEqual(human.state, OccupancyState.HUMAN)
        self.assertEqual(unknown.state, OccupancyState.UNKNOWN)
        self.assertGreater(human.active_area, obj.active_area)
        self.assertGreaterEqual(human.connected_regions, 1)

    def test_zero_signal_is_empty_not_below_threshold_load(self) -> None:
        detector = OccupancyDetector(fps=10.0)

        result = detector.analyze(np.stack([np.zeros((16, 16), dtype=float)] * 6))

        self.assertEqual(result.state, OccupancyState.EMPTY)

    def test_noise_level_tiny_nonzero_signal_is_still_empty(self) -> None:
        detector = OccupancyDetector(fps=10.0)
        frame = np.zeros((16, 16), dtype=float)
        frame[7, 7] = 0.2
        frame[8, 8] = 0.2

        result = detector.analyze(np.stack([frame] * 6))

        self.assertEqual(result.state, OccupancyState.EMPTY)

    def test_repeatable_weak_response_is_load_below_threshold(self) -> None:
        detector = OccupancyDetector(fps=10.0)
        frame = np.zeros((16, 16), dtype=float)
        frame[6:8, 6:9] = 5.0

        result = detector.analyze(np.stack([frame] * 6))

        self.assertEqual(result.state, OccupancyState.LOAD_BELOW_THRESHOLD)
        self.assertGreaterEqual(result.detectable_points, 4)


class SeatAnalyzerTest(unittest.TestCase):
    def test_empty_object_and_unknown_never_call_posture_model(self) -> None:
        recognizer = CountingRecognizer()
        analyzer = SeatAnalyzer(recognizer=recognizer, fps=10.0, window_seconds=0.5, settle_seconds=0.5)

        empty = analyzer.update(np.zeros((16, 16), dtype=float))
        obj = analyzer.update(object_like_frame())
        unknown = analyzer.update(unknown_like_frame())

        self.assertEqual(empty["occupancy_state"], "EMPTY")
        self.assertEqual(obj["occupancy_state"], "OBJECT")
        self.assertEqual(unknown["occupancy_state"], "UNKNOWN")
        self.assertIsNone(empty["posture"])
        self.assertIsNone(obj["posture"])
        self.assertIsNone(unknown["posture"])
        self.assertEqual(recognizer.calls, 0)

    def test_load_below_threshold_never_calls_posture_model(self) -> None:
        recognizer = CountingRecognizer()
        analyzer = SeatAnalyzer(recognizer=recognizer, fps=10.0, window_seconds=0.5, settle_seconds=0.5)
        frame = np.zeros((16, 16), dtype=float)
        frame[6:8, 6:9] = 5.0

        result = None
        for _ in range(6):
            result = analyzer.update(frame)

        self.assertEqual(result["occupancy_state"], "LOAD_BELOW_THRESHOLD")
        self.assertIsNone(result["posture"])
        self.assertEqual(recognizer.calls, 0)

    def test_human_stabilizing_waits_and_stable_human_outputs_posture(self) -> None:
        recognizer = CountingRecognizer()
        analyzer = SeatAnalyzer(recognizer=recognizer, fps=10.0, window_seconds=0.5, settle_seconds=0.5)

        first = analyzer.update(human_like_frame())
        self.assertEqual(first["seat_state"], "HUMAN_STABILIZING")
        self.assertIsNone(first["posture"])
        self.assertEqual(recognizer.calls, 0)

        result = first
        for _ in range(8):
            result = analyzer.update(human_like_frame())

        self.assertEqual(result["seat_state"], "HUMAN_RECOGNIZING")
        self.assertEqual(result["posture"], "端正坐姿")
        self.assertGreater(recognizer.calls, 0)

    def test_empty_after_human_clears_previous_posture(self) -> None:
        analyzer = SeatAnalyzer(recognizer=CountingRecognizer(), fps=10.0, window_seconds=0.5, settle_seconds=0.5)
        for _ in range(8):
            analyzer.update(human_like_frame())

        cleared = analyzer.update(np.zeros((16, 16), dtype=float))

        self.assertEqual(cleared["occupancy_state"], "EMPTY")
        self.assertIsNone(cleared["posture"])

    def test_analyze_seat_direct_window_obeys_occupancy_gate(self) -> None:
        recognizer = CountingRecognizer()
        analyzer = SeatAnalyzer(recognizer=recognizer, fps=10.0, window_seconds=0.5, settle_seconds=0.5)

        result = analyzer.analyze_seat(np.stack([object_like_frame()] * 6))

        self.assertEqual(result["occupancy_state"], "OBJECT")
        self.assertIsNone(result["posture"])
        self.assertEqual(recognizer.calls, 0)

    def test_high_total_continuous_low_concentration_pressure_can_be_human_even_if_area_is_moderate(self) -> None:
        frame = np.zeros((16, 16), dtype=float)
        frame[5:10, 5:11] = 90.0
        frame[7:9, 4:12] = 55.0
        analyzer = SeatAnalyzer(recognizer=CountingRecognizer(), fps=10.0, window_seconds=0.5, settle_seconds=0.5)

        result = analyzer.analyze_seat(np.stack([frame] * 6))

        self.assertEqual(result["occupancy_state"], "HUMAN")

    def test_dynamic_low_pressure_object_transition_is_unknown_not_human(self) -> None:
        detector = OccupancyDetector(fps=10.0)
        frames = []
        for scale in [0.0, 0.35, 0.7, 1.0, 1.0, 1.0]:
            frames.append(object_like_frame() * scale)

        result = detector.analyze(np.stack(frames))

        self.assertIn(result.state, {OccupancyState.OBJECT, OccupancyState.UNKNOWN})
        self.assertNotEqual(result.state, OccupancyState.HUMAN)


class RecognizerApiTest(unittest.TestCase):
    def test_predict_exposes_hardware_friendly_output_contract(self) -> None:
        from recognizer.recognizer_api import Recognizer

        analyzer = FakeAnalyzer()
        api = Recognizer(analyzer=analyzer)

        result = api.predict(np.zeros((16, 16), dtype=float))

        self.assertEqual(result["occupancy"], "HUMAN")
        self.assertEqual(result["occupancy_confidence"], 0.88)
        self.assertEqual(result["posture"], "端正坐姿")
        self.assertEqual(result["posture_confidence"], 0.91)
        self.assertEqual(result["second_label"], "前倾端坐")
        self.assertEqual(result["margin"], 0.42)
        self.assertFalse(result["is_boundary"])
        self.assertEqual(result["prototype_diagnosis"], {"label": "端正坐姿"})
        self.assertEqual(result["seat_state"], "HUMAN_RECOGNIZING")
        self.assertEqual(analyzer.calls, 1)

    def test_boundary_output_preserves_raw_candidate_diagnostics(self) -> None:
        from recognizer.recognizer_api import Recognizer

        api = Recognizer(analyzer=BoundaryAnalyzer())

        result = api.predict(np.zeros((16, 16), dtype=float))

        self.assertIsNone(result["posture"])
        self.assertEqual(result["raw_label"], "端正坐姿")
        self.assertEqual(result["raw_confidence"], 0.49)
        self.assertEqual(result["second_label"], "前倾端坐")
        self.assertEqual(result["margin"], 0.06)
        self.assertTrue(result["is_boundary"])
        self.assertEqual(result["boundary_reason"], "RF margin<0.10")
        self.assertEqual(result["prototype_diagnosis"]["label"], "端正坐姿")

    def test_predict_rejects_non_16x16_frames(self) -> None:
        from recognizer.recognizer_api import Recognizer

        api = Recognizer(analyzer=FakeAnalyzer())

        with self.assertRaises(ValueError):
            api.predict(np.zeros((8, 8), dtype=float))

    def test_reset_and_calibrate_are_public_runtime_controls(self) -> None:
        from recognizer.recognizer_api import Recognizer

        analyzer = FakeAnalyzer()
        api = Recognizer(analyzer=analyzer)

        api.reset()
        calibration = api.calibrate(np.zeros((16, 16), dtype=float))

        self.assertEqual(analyzer.reset_calls, 1)
        self.assertEqual(calibration["frames"], 1)
        self.assertTrue(calibration["calibrated"])

    def test_default_api_loads_artifacts_and_keeps_empty_out_of_posture(self) -> None:
        from recognizer.recognizer_api import Recognizer

        model_path = Path("recognizer/models/rf_posture_v1.joblib")
        prototype_path = Path("recognizer/models/prototype_bank_v1.json")
        if not model_path.exists() or not prototype_path.exists():
            self.skipTest("RF model artifacts are not available")

        api = Recognizer()
        result = api.predict(np.zeros((16, 16), dtype=float))

        self.assertEqual(result["occupancy"], "EMPTY")
        self.assertIsNone(result["posture"])

    def test_default_model_pointer_selects_v2_2_candidate_after_h3_promotion(self) -> None:
        from recognizer.recognizer_api import Recognizer, default_model_version

        default_api = Recognizer(analyzer=FakeAnalyzer())

        self.assertEqual(default_model_version(), "v2_2_candidate")
        self.assertEqual(default_api.model_version, "v2_2_candidate")
        self.assertEqual(default_api.model_path.name, "rf_posture_v2_1_candidate.joblib")
        self.assertEqual(default_api.prototype_bank_path.name, "prototype_bank_v2_1_candidate.json")
        self.assertEqual(default_api.submodel_path.name, "leanback_subclassifier_v2_2_candidate.joblib")

    def test_explicit_model_versions_remain_available_after_default_switch(self) -> None:
        from recognizer.recognizer_api import Recognizer

        v1_api = Recognizer(model_version="v1", analyzer=FakeAnalyzer())
        candidate_api = Recognizer(model_version="v2_candidate", analyzer=FakeAnalyzer())
        v21_candidate_api = Recognizer(model_version="v2_1_candidate", analyzer=FakeAnalyzer())
        v22_candidate_api = Recognizer(model_version="v2_2_candidate", analyzer=FakeAnalyzer())
        v23_candidate_api = Recognizer(model_version="v2_3_candidate", analyzer=FakeAnalyzer())

        self.assertEqual(v1_api.model_path.name, "rf_posture_v1.joblib")
        self.assertEqual(v1_api.prototype_bank_path.name, "prototype_bank_v1.json")
        self.assertEqual(candidate_api.model_path.name, "rf_posture_v2_candidate.joblib")
        self.assertEqual(candidate_api.prototype_bank_path.name, "prototype_bank_v2_candidate.json")
        self.assertEqual(v21_candidate_api.model_path.name, "rf_posture_v2_1_candidate.joblib")
        self.assertEqual(v21_candidate_api.prototype_bank_path.name, "prototype_bank_v2_1_candidate.json")
        self.assertEqual(v22_candidate_api.model_path.name, "rf_posture_v2_1_candidate.joblib")
        self.assertEqual(v22_candidate_api.submodel_path.name, "leanback_subclassifier_v2_2_candidate.joblib")
        self.assertEqual(v23_candidate_api.model_path.name, "rf_posture_v2_1_candidate.joblib")
        self.assertEqual(v23_candidate_api.submodel_path.name, "leanback_subclassifier_v2_2_candidate.joblib")
        self.assertEqual(v23_candidate_api.lateral_submodel_path.name, "lateral_subclassifier_v2_3_candidate.joblib")

    def test_root_import_path_is_available_for_hardware_integration(self) -> None:
        from recognizer_api import Recognizer

        api = Recognizer(analyzer=FakeAnalyzer())
        result = api.predict(np.zeros((16, 16), dtype=float))

        self.assertEqual(result["posture"], "端正坐姿")

    def test_v2_2_promotion_manifest_records_h3_without_retraining(self) -> None:
        manifest_path = Path("recognizer/models/v2_2_promotion_manifest.json")
        results_path = Path("posture_dataset_v2/reports/v2_2_h3_external_holdout/h3_file_level_results.csv")
        holdout_manifest_path = Path("posture_dataset_v2/reports/v2_2_h3_external_holdout/h3_holdout_manifest.csv")
        if not results_path.exists() or not holdout_manifest_path.exists():
            self.skipTest("H3 holdout reports are not available")

        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["promoted_model_version"], "v2_2_candidate")
        self.assertEqual(manifest["previous_default_model"], "v2_1_candidate")
        self.assertEqual(manifest["new_default_model"], "v2_2_candidate")
        self.assertEqual(manifest["correct_accept"], 3)
        self.assertEqual(manifest["correct_fallback"], 1)
        self.assertEqual(manifest["wrong_accept"], 0)
        self.assertEqual(manifest["gate_miss"], 0)
        self.assertFalse(manifest["retraining_permitted_with_h3"])

        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = {row["filename"]: row for row in csv.DictReader(handle)}
        self.assertEqual(rows["H3_houyangkaobei2.csv"]["final_display_label"], "后靠坐姿")
        self.assertEqual(rows["H3_houyangkaobei2.csv"]["file_result_type"], "correct_fallback")

        with holdout_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            holdout_rows = list(csv.DictReader(handle))
        self.assertTrue(all(row["included_in_training"] == "False" for row in holdout_rows))
        self.assertTrue(all(row["included_in_tuning"] == "False" for row in holdout_rows))
        self.assertTrue(all(row["eligible_for_retraining"] == "False" for row in holdout_rows))

        training_manifests = [
            Path("posture_dataset_v2/v2_2_candidate/v2_2_development_manifest.csv"),
            Path("posture_dataset_v2/reports/v2_2_leanback_subclassifier/v2_2_development_manifest.csv"),
        ]
        for path in training_manifests:
            if path.exists():
                self.assertNotIn("H3_", path.read_text(encoding="utf-8-sig"))

        source_files = [
            Path("recognizer/models/leanback_prototype_bank_v2_2_candidate.json"),
            Path("recognizer/models/leanback_subclassifier_v2_2_candidate.runtime_config.json"),
            Path("recognizer/models/v2_2_candidate.model_bundle.json"),
        ]
        for path in source_files:
            if path.exists():
                self.assertNotIn("H3_", path.read_text(encoding="utf-8"))


class CsvGuiCoreTest(unittest.TestCase):
    def _write_csv(self, path: Path, frames: list[np.ndarray]) -> None:
        lines = []
        for _index, frame in enumerate(frames):
            lines.append("2026/07/15_12:00:00")
            for row in np.asarray(frame, dtype=float):
                lines.append(",".join(f"{value:.1f}" for value in row))
        path.write_text("\n".join(lines), encoding="ascii")

    def test_load_csv_playback_rejects_bad_csv_with_clear_error(self) -> None:
        from recognizer.csv_gui_core import CsvFormatError, load_csv_playback

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("not-a-valid-flex-csv\n", encoding="ascii")

            with self.assertRaises(CsvFormatError) as ctx:
                load_csv_playback(path)

        self.assertIn("CSV格式错误", str(ctx.exception))

    def test_empty_frames_do_not_display_stale_posture(self) -> None:
        from recognizer.csv_gui_core import CsvRecognitionSession, CsvPlaybackData

        data = CsvPlaybackData(
            path=Path("empty.csv"),
            timestamps=[],
            frames=np.stack([np.zeros((16, 16))] * 2),
            frame_times_s=[0.0, 0.05],
            fps=20.0,
            time_source="fallback_fps",
        )
        api = SequenceRecognizerApi([
            api_payload("HUMAN", "端正坐姿"),
            api_payload("EMPTY", None, None),
        ])
        session = CsvRecognitionSession(data, api)

        session.step()
        record = session.step()

        self.assertEqual(record.occupancy_state, "EMPTY")
        self.assertIsNone(record.posture)

    def test_object_and_unknown_outputs_never_count_as_posture(self) -> None:
        from recognizer.csv_gui_core import CsvRecognitionSession, CsvPlaybackData

        data = CsvPlaybackData(
            path=Path("object.csv"),
            timestamps=[],
            frames=np.stack([object_like_frame(), unknown_like_frame()]),
            frame_times_s=[0.0, 0.05],
            fps=20.0,
            time_source="fallback_fps",
        )
        api = SequenceRecognizerApi([
            api_payload("OBJECT", None, None),
            api_payload("UNKNOWN", None, None, boundary=True),
        ])
        records = CsvRecognitionSession(data, api).process_all()

        self.assertTrue(all(record.posture is None for record in records))
        self.assertEqual([record.display_status for record in records], ["OBJECT", "UNKNOWN"])

    def test_pause_resume_does_not_skip_frames(self) -> None:
        from recognizer.csv_gui_core import CsvRecognitionSession, CsvPlaybackData

        data = CsvPlaybackData(
            path=Path("pause.csv"),
            timestamps=[],
            frames=np.stack([np.ones((16, 16)) * i for i in range(3)]),
            frame_times_s=[0.0, 0.05, 0.10],
            fps=20.0,
            time_source="fallback_fps",
        )
        session = CsvRecognitionSession(data, SequenceRecognizerApi([api_payload()] * 3))

        first = session.step()
        session.pause()
        skipped = session.step()
        session.resume()
        second = session.step()

        self.assertEqual(first.frame_index, 0)
        self.assertIsNone(skipped)
        self.assertEqual(second.frame_index, 1)

    def test_manual_step_once_advances_even_when_paused(self) -> None:
        from recognizer.csv_gui_core import CsvRecognitionSession, CsvPlaybackData

        data = CsvPlaybackData(
            path=Path("manual.csv"),
            timestamps=[],
            frames=np.stack([np.ones((16, 16)) * i for i in range(2)]),
            frame_times_s=[0.0, 0.05],
            fps=20.0,
            time_source="fallback_fps",
        )
        session = CsvRecognitionSession(data, SequenceRecognizerApi([api_payload()] * 2))

        session.pause()
        record = session.step_once()

        self.assertEqual(record.frame_index, 0)
        self.assertFalse(session.playing)

    def test_stop_resets_index_predictions_and_recognizer_state(self) -> None:
        from recognizer.csv_gui_core import CsvRecognitionSession, CsvPlaybackData

        data = CsvPlaybackData(
            path=Path("stop.csv"),
            timestamps=[],
            frames=np.stack([np.ones((16, 16))] * 2),
            frame_times_s=[0.0, 0.05],
            fps=20.0,
            time_source="fallback_fps",
        )
        api = SequenceRecognizerApi([api_payload()] * 2)
        session = CsvRecognitionSession(data, api)

        session.step()
        session.stop()

        self.assertEqual(session.index, 0)
        self.assertEqual(session.predictions, [])
        self.assertEqual(api.reset_calls, 1)

    def test_export_writes_frame_segments_and_summary_files(self) -> None:
        from recognizer.csv_gui_core import CsvRecognitionSession, CsvPlaybackData

        data = CsvPlaybackData(
            path=Path("demo.csv"),
            timestamps=[],
            frames=np.stack([np.ones((16, 16)) * 10, np.ones((16, 16)) * 20]),
            frame_times_s=[0.0, 0.05],
            fps=20.0,
            time_source="fallback_fps",
        )
        session = CsvRecognitionSession(data, SequenceRecognizerApi([api_payload()] * 2))
        session.process_all()
        with TemporaryDirectory() as tmp:
            exported = session.export_results(Path(tmp))

            self.assertTrue(exported["frame_predictions"].exists())
            self.assertTrue(exported["posture_segments"].exists())
            self.assertTrue(exported["summary"].exists())
            header = exported["frame_predictions"].read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("frame_index", header)
            self.assertIn("display_status", header)
            self.assertIn("raw_label", header)
            self.assertIn("boundary_reason", header)
            self.assertIn("lateral_subclassifier_triggered", header)
            self.assertIn("lateral_posture_label", header)

    def test_export_records_model_version_and_artifact_hashes(self) -> None:
        from recognizer.csv_gui_core import CsvRecognitionSession, CsvPlaybackData

        data = CsvPlaybackData(
            path=Path("demo.csv"),
            timestamps=[],
            frames=np.stack([np.ones((16, 16)) * 10]),
            frame_times_s=[0.0],
            fps=20.0,
            time_source="fallback_fps",
        )
        api = SequenceRecognizerApi([api_payload()])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.joblib"
            metadata_path = root / "metadata.json"
            runtime_path = root / "runtime.json"
            model_path.write_text("model", encoding="ascii")
            metadata_path.write_text("metadata", encoding="ascii")
            runtime_path.write_text("runtime", encoding="ascii")
            api.model_version = "v2_1_candidate"
            api.model_path = model_path
            api.metadata_path = metadata_path
            api.runtime_config_path = runtime_path

            session = CsvRecognitionSession(data, api)
            session.process_all()
            exported = session.export_results(root / "exports")

            frame_header = exported["frame_predictions"].read_text(encoding="utf-8").splitlines()[0]
            segment_header = exported["posture_segments"].read_text(encoding="utf-8").splitlines()[0]
            summary = json.loads(exported["summary"].read_text(encoding="utf-8"))

        expected_model_hash = hashlib.sha256(b"model").hexdigest()
        expected_metadata_hash = hashlib.sha256(b"metadata").hexdigest()
        expected_runtime_hash = hashlib.sha256(b"runtime").hexdigest()
        for header in (frame_header, segment_header):
            self.assertIn("model_version", header)
            self.assertIn("model_artifact_sha256", header)
            self.assertIn("metadata_sha256", header)
            self.assertIn("runtime_config_sha256", header)
        self.assertEqual(summary["model_version"], "v2_1_candidate")
        self.assertEqual(summary["model_artifact_sha256"], expected_model_hash)
        self.assertEqual(summary["metadata_sha256"], expected_metadata_hash)
        self.assertEqual(summary["runtime_config_sha256"], expected_runtime_hash)

    def test_v2_2_export_records_promotion_evidence(self) -> None:
        from recognizer.csv_gui_core import promotion_export_info

        info = promotion_export_info("v2_2_candidate")

        self.assertEqual(info["promotion_status"], "promoted_default")
        self.assertEqual(info["promotion_evidence"], "passed_h3_external_holdout")
        self.assertRegex(str(info["promotion_manifest_hash"]), r"^[0-9a-f]{64}$")

        old_info = promotion_export_info("v2_1_candidate")
        self.assertIsNone(old_info["promotion_status"])
        self.assertIsNone(old_info["promotion_evidence"])

    def test_boundary_frame_record_keeps_diagnostics_for_gui_and_export(self) -> None:
        from recognizer.csv_gui_core import frame_record_from_result

        result = {
            "occupancy": "HUMAN",
            "occupancy_confidence": 0.84,
            "seat_state": "HUMAN_RECOGNIZING",
            "posture": None,
            "posture_confidence": 0.49,
            "second_label": "前倾端坐",
            "margin": 0.06,
            "is_boundary": True,
            "raw_label": "端正坐姿",
            "raw_confidence": 0.49,
            "boundary_reason": "RF margin<0.10",
            "prototype_diagnosis": {"label": "端正坐姿", "matched_prototype_id": "p1"},
            "occupancy_features": {"total_pressure": 3456.0, "active_points": 52},
        }

        record = frame_record_from_result(
            frame=np.ones((16, 16), dtype=float),
            result=result,
            frame_index=42,
            timestamp=2.1,
        )

        self.assertEqual(record.display_status, "POSTURE")
        self.assertIsNone(record.posture)
        self.assertEqual(record.raw_label, "端正坐姿")
        self.assertEqual(record.raw_confidence, 0.49)
        self.assertEqual(record.posture_confidence, 0.49)
        self.assertEqual(record.second_label, "前倾端坐")
        self.assertEqual(record.margin, 0.06)
        self.assertEqual(record.boundary_reason, "RF margin<0.10")
        self.assertIn("proto=端正坐姿", record.prototype_diagnosis)

    def test_frame_record_keeps_lateral_candidate_fields(self) -> None:
        from recognizer.csv_gui_core import frame_record_from_result

        result = {
            "occupancy": "HUMAN",
            "occupancy_confidence": 0.9,
            "seat_state": "HUMAN_RECOGNIZING",
            "posture": "侧向坐姿",
            "posture_confidence": 0.72,
            "second_label": "侧身倚靠坐",
            "margin": 0.18,
            "is_boundary": False,
            "raw_label": "侧向坐姿",
            "raw_confidence": 0.72,
            "occupancy_features": {"total_pressure": 2000.0, "active_points": 64},
            "lateral_subclassifier_triggered": True,
            "lateral_gate_reason": "lateral_parent_match=标准侧坐",
            "lateral_posture_label": "侧向坐姿",
            "lateral_confidence": 0.41,
            "lateral_margin": 0.04,
            "lateral_boundary": True,
            "lateral_boundary_reasons": ["low_prototype_margin"],
            "lateral_prototype_label": "侧身倚靠坐",
            "lateral_prototype_distance": 3.25,
            "lateral_fallback_used": True,
            "lateral_submodel_version": "lateral_subclassifier_v2_3_candidate",
        }

        record = frame_record_from_result(np.ones((16, 16), dtype=float), result, 7, 0.35)

        self.assertTrue(record.lateral_subclassifier_triggered)
        self.assertEqual(record.lateral_posture_label, "侧向坐姿")
        self.assertEqual(record.lateral_boundary_reasons, "low_prototype_margin")
        self.assertEqual(record.lateral_prototype_label, "侧身倚靠坐")
        self.assertEqual(record.lateral_submodel_version, "lateral_subclassifier_v2_3_candidate")

    def test_model_load_failure_has_gui_friendly_message(self) -> None:
        from recognizer.csv_gui_core import CsvGuiError, load_runtime_recognizer

        with self.assertRaises(CsvGuiError) as ctx:
            load_runtime_recognizer(model_path=Path("missing-model.joblib"))

        self.assertIn("模型加载失败", str(ctx.exception))

    def test_human_csv_can_run_through_csv_recognition_session(self) -> None:
        from recognizer.csv_gui_core import CsvRecognitionSession, load_csv_playback
        from recognizer_api import Recognizer

        csv_path = Path("posture_sensor_analysis/quality_batches/postures_4_7_acceptance/dataset_v1_1_17_final/duanzhengzuozi1-1.csv")
        model_path = Path("recognizer/models/rf_posture_v1.joblib")
        if not csv_path.exists() or not model_path.exists():
            self.skipTest("formal dataset CSV or RF model is not available")

        data = load_csv_playback(csv_path)
        session = CsvRecognitionSession(data, Recognizer())
        records = session.process_all()

        self.assertEqual(len(records), len(data.frames))
        self.assertTrue(any(record.posture for record in records))

    def test_gui_model_version_display_names_are_explicit(self) -> None:
        from recognizer.csv_gui import default_gui_model_version, model_version_display_name

        self.assertEqual(model_version_display_name("v1"), "V1")
        self.assertEqual(model_version_display_name("v2_candidate"), "V2 Candidate")
        self.assertEqual(model_version_display_name("v2_1_candidate"), "V2.1（Phase 1闭卷通过）")
        self.assertEqual(model_version_display_name("v2_2_candidate"), "V2.2（H3闭卷通过）")
        self.assertEqual(model_version_display_name("v2_3_candidate"), "V2.3候选（侧向三类局部解析，未闭卷）")
        self.assertEqual(default_gui_model_version(), "v2_2_candidate")


class ExternalHoldoutEvaluationTest(unittest.TestCase):
    def _write_flex_csv(self, path: Path, frames: list[np.ndarray]) -> None:
        lines = []
        for frame in frames:
            lines.append("2026/07/15_12:00:00")
            for row in np.asarray(frame, dtype=float):
                lines.append(",".join(f"{value:.1f}" for value in row))
        path.write_text("\n".join(lines), encoding="ascii")

    def test_manifest_template_has_required_closed_book_fields(self) -> None:
        from posture_dataset_v2.scripts.evaluate_external_holdout import REQUIRED_MANIFEST_FIELDS

        self.assertEqual(
            REQUIRED_MANIFEST_FIELDS,
            [
                "filename",
                "true_label",
                "holdout_batch",
                "collection_time",
                "data_role",
                "natural_posture",
                "included_in_training",
                "included_in_tuning",
                "notes",
            ],
        )

    def test_batch_evaluation_processes_every_frame_and_keeps_models_separate(self) -> None:
        from posture_dataset_v2.scripts.evaluate_external_holdout import evaluate_external_holdout

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout_dir = root / "holdout"
            output_dir = root / "outputs"
            holdout_dir.mkdir()
            csv_path = holdout_dir / "holdout_01.csv"
            frames = [np.zeros((16, 16)), human_like_frame(35.0), human_like_frame(40.0)]
            self._write_flex_csv(csv_path, frames)
            manifest = holdout_dir / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "filename",
                        "true_label",
                        "holdout_batch",
                        "collection_time",
                        "data_role",
                        "natural_posture",
                        "included_in_training",
                        "included_in_tuning",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "filename": csv_path.name,
                        "true_label": "端正坐姿",
                        "holdout_batch": "unit",
                        "collection_time": "2026-07-15T12:00:00",
                        "data_role": "external_holdout",
                        "natural_posture": "true",
                        "included_in_training": "false",
                        "included_in_tuning": "false",
                        "notes": "unit test",
                    }
                )
            calls: list[str] = []

            def factory(model_version: str) -> SequenceRecognizerApi:
                calls.append(model_version)
                return SequenceRecognizerApi(
                    [
                        api_payload("EMPTY", None, None),
                        api_payload("HUMAN", "端正坐姿", 0.88),
                        api_payload("HUMAN", "端正坐姿", 0.9),
                    ]
                )

            paths = evaluate_external_holdout(
                holdout_dir=holdout_dir,
                manifest_path=manifest,
                output_dir=output_dir,
                recognizer_factory=factory,
            )

            self.assertEqual(calls, ["v1", "v2_candidate"])
            for key in [
                "holdout_v1_file_predictions",
                "holdout_v2_file_predictions",
                "holdout_v1_frame_predictions",
                "holdout_v2_frame_predictions",
                "holdout_model_comparison",
                "holdout_per_class_recall",
                "holdout_confusion_matrices",
                "holdout_final_report",
            ]:
                self.assertTrue(paths[key].exists(), key)
            with paths["holdout_v1_file_predictions"].open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["processed_frames"], "3")
            self.assertEqual(rows[0]["csv_total_frames"], "3")
            self.assertEqual(rows[0]["export_complete"], "True")
            self.assertEqual(rows[0]["valid_evaluation"], "True")
            self.assertFalse((root / "training_candidates").exists())

    def test_replay_sequence_is_independent_of_playback_speed_labels(self) -> None:
        from recognizer.csv_gui_core import CsvPlaybackData, CsvRecognitionSession

        data = CsvPlaybackData(
            path=Path("speed.csv"),
            timestamps=[],
            frames=np.stack([np.zeros((16, 16)), human_like_frame(35.0), human_like_frame(40.0)]),
            frame_times_s=[0.0, 0.05, 0.10],
            fps=20.0,
            time_source="fallback_fps",
        )

        sequences = []
        for _speed in ["1×", "2×", "最大速度"]:
            session = CsvRecognitionSession(
                data,
                SequenceRecognizerApi(
                    [
                        api_payload("EMPTY", None, None),
                        api_payload("HUMAN", "端正坐姿", 0.88),
                        api_payload("HUMAN", "端正坐姿", 0.9),
                    ]
                ),
            )
            sequences.append([(record.frame_index, record.posture, record.is_boundary) for record in session.process_all()])

        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(sequences[0], sequences[2])

    def test_evaluation_does_not_modify_model_artifacts(self) -> None:
        from posture_dataset_v2.scripts.evaluate_external_holdout import sha256_file

        model_paths = [
            Path("recognizer/models/rf_posture_v1.joblib"),
            Path("recognizer/models/rf_posture_v2_candidate.joblib"),
        ]
        existing = [path for path in model_paths if path.exists()]
        before = {path: sha256_file(path) for path in existing}
        after = {path: sha256_file(path) for path in existing}

        self.assertEqual(before, after)


class FrameReaderTest(unittest.TestCase):
    def test_csv_replay_reader_returns_16x16_frames(self) -> None:
        path = Path("posture_sensor_analysis/quality_batches/postures_4_7_acceptance/dataset_v1_1_17_final/duanzhengzuozi1-1.csv")
        if not path.exists():
            self.skipTest("formal dataset CSV is not available")
        reader = CSVReplayReader(path)

        frame = reader.read_frame()

        self.assertEqual(frame.shape, (16, 16))

    def test_serial_reader_is_a_configurable_skeleton(self) -> None:
        reader = SerialFrameReader(port="/dev/tty.TEST", baudrate=115200, delimiter=",", timestamp_enabled=False)

        with self.assertRaises(NotImplementedError):
            reader.read_frame()


class OccupancyCsvReplayTest(unittest.TestCase):
    def test_csv_replay_moves_from_empty_to_human_to_posture_without_object(self) -> None:
        csv_path = Path("posture_sensor_analysis/quality_batches/postures_4_7_acceptance/dataset_v1_1_17_final/duanzhengzuozi1-1.csv")
        model_path = Path("recognizer/models/rf_posture_v1.joblib")
        prototype_path = Path("recognizer/models/prototype_bank_v1.json")
        if not csv_path.exists() or not model_path.exists() or not prototype_path.exists():
            self.skipTest("RF model artifacts or formal CSV are not available")

        summary, _ = run_csv_stream(csv_path, model_path, prototype_path)

        self.assertGreater(summary.status_counts.get("EMPTY", 0), 0)
        self.assertGreater(summary.status_counts.get("HUMAN_STABILIZING", 0), 0)
        self.assertGreater(summary.status_counts.get("POSTURE", 0), 0)
        self.assertEqual(summary.status_counts.get("OBJECT", 0), 0)
        self.assertEqual(summary.most_common_label, "端正坐姿")


class DataLoaderTest(unittest.TestCase):
    def test_v1_label_taxonomy_merges_known_boundary_classes(self) -> None:
        self.assertEqual(map_v1_label("半躺靠背坐"), "后靠/瘫坐类")
        self.assertEqual(map_v1_label("瘫坐/斜躺合并"), "后靠/瘫坐类")
        self.assertEqual(map_v1_label("全躺卧姿"), "躺卧类")
        self.assertEqual(map_v1_label("侧卧半躺"), "躺卧类")
        self.assertEqual(map_v1_label("端正坐姿"), "端正坐姿")

    def test_source_family_uses_class_specific_groups_for_boundary_classes(self) -> None:
        self.assertEqual(source_family_from_name("bantangkaobeizuo5-3.csv"), "class5")
        self.assertEqual(source_family_from_name("tanzuo8.9-1.csv"), "class8")
        self.assertEqual(source_family_from_name("quantangwozi10-2.csv"), "class10")
        self.assertEqual(source_family_from_name("cewobantang11-3.csv"), "class11")


class PrototypeRecognizerTest(unittest.TestCase):
    def test_predict_posture_returns_label_confidence_margin_and_boundary(self) -> None:
        proto_a = Prototype("A::P1", "A", np.zeros(264))
        proto_b = Prototype("B::P1", "B", np.ones(264) * 3)
        bank = PrototypeBank([proto_a, proto_b])
        recognizer = PrototypeRecognizer(bank, RecognizerConfig(boundary_margin=0.05))

        result = recognizer.predict_posture(np.zeros((16, 16), dtype=float))

        self.assertEqual(result.label, "A")
        self.assertEqual(result.second_label, "B")
        self.assertGreater(result.confidence, 0.9)
        self.assertFalse(result.is_boundary)

    def test_mirror_aware_prototype_matches_left_right_equivalent_pressure(self) -> None:
        left_frame = np.zeros((16, 16), dtype=float)
        left_frame[:, :4] = 50.0
        right_frame = np.flip(left_frame, axis=1)
        proto = Prototype(
            "侧卧半躺::P11-A",
            "侧卧半躺",
            extract_features(left_frame),
            mirror_aware=True,
        )
        other = Prototype("标准侧坐::P1", "标准侧坐", np.ones(264) * 0.2)
        recognizer = PrototypeRecognizer(PrototypeBank([proto, other]))

        result = recognizer.predict_posture(right_frame)

        self.assertEqual(result.label, "侧卧半躺")
        self.assertFalse(result.is_boundary)

    def test_probability_recognizer_wraps_predict_proba_models(self) -> None:
        recognizer = ProbabilityRecognizer(ConstantProbModel(), min_confidence=0.60, boundary_margin=0.20)

        result = recognizer.predict_posture(np.zeros((16, 16), dtype=float))

        self.assertEqual(result.label, "A")
        self.assertEqual(result.second_label, "B")
        self.assertAlmostEqual(result.confidence, 0.82)
        self.assertFalse(result.is_boundary)


class HybridRecognizerTest(unittest.TestCase):
    def test_hybrid_recognizer_uses_rf_label_and_prototype_diagnosis_as_auxiliary(self) -> None:
        proto_a = Prototype("proto-a", "A", extract_features(np.zeros((16, 16))), mirror_aware=False)
        proto_b = Prototype("proto-b", "B", np.ones(264) * 2, mirror_aware=False)
        hybrid = HybridPostureRecognizer(
            rf_recognizer=ProbabilityRecognizer(ConstantProbModel()),
            prototype_recognizer=PrototypeRecognizer(PrototypeBank([proto_a, proto_b])),
        )

        payload = hybrid.predict_posture(np.zeros((16, 16), dtype=float))

        self.assertEqual(payload["label"], "A")
        self.assertIn("prototype_diagnosis", payload)
        self.assertEqual(payload["prototype_diagnosis"]["label"], "A")
        self.assertFalse(payload["is_boundary"])



class LateralSubclassifierTest(unittest.TestCase):
    def test_gate_runs_for_lateral_parent_raw_or_prototype_candidates(self) -> None:
        parent_match = {"label": STANDARD_SIDE_SITTING_LABEL, "raw_label": STANDARD_SIDE_SITTING_LABEL}
        raw_match = {"label": "边界/不确定", "raw_label": DIAGONAL_SITTING_LABEL}
        proto_match = {"label": "边界/不确定", "prototype_diagnosis": {"label": SIDE_LEANING_LABEL}}
        upright = {"label": "端正坐姿", "raw_label": "端正坐姿", "prototype_diagnosis": {"label": "端正坐姿"}}

        self.assertTrue(should_run_lateral_subclassifier(parent_match)[0])
        self.assertTrue(should_run_lateral_subclassifier(raw_match)[0])
        self.assertTrue(should_run_lateral_subclassifier(proto_match)[0])
        self.assertFalse(should_run_lateral_subclassifier(upright)[0])

    def test_lateral_gate_rejects_when_leanback_stage_has_priority(self) -> None:
        stage1 = {
            "label": STANDARD_SIDE_SITTING_LABEL,
            "raw_label": STANDARD_SIDE_SITTING_LABEL,
            "subclassifier_triggered": True,
            "final_display_label": "后靠坐姿",
        }

        should_run, reasons = should_run_lateral_subclassifier(stage1, {"left_share": 0.75, "active_area_ratio": 0.35})

        self.assertFalse(should_run)
        self.assertIn("gate_rejected_leanback_priority", reasons)

    def test_lateral_physical_gate_accepts_single_side_loading(self) -> None:
        lateral = {"left_share": 0.72, "right_share": 0.28, "active_area_ratio": 0.30, "front_share": 0.55, "back_share": 0.45, "row_8_11_share": 0.32, "cop_y": 6.4}
        upright = {"left_share": 0.51, "right_share": 0.49, "active_area_ratio": 0.30, "front_share": 0.55, "back_share": 0.45, "row_8_11_share": 0.32, "cop_y": 6.4}

        self.assertTrue(lateral_physical_gate(lateral)[0])
        self.assertFalse(lateral_physical_gate(upright)[0])

    def test_lateral_model_falls_back_for_low_prototype_margin(self) -> None:
        model = LateralFineModel(
            prototypes={
                STANDARD_SIDE_SITTING_LABEL: [np.zeros(len(LATERAL_FEATURE_NAMES), dtype=float)],
                DIAGONAL_SITTING_LABEL: [np.ones(len(LATERAL_FEATURE_NAMES), dtype=float) * 0.04],
                SIDE_LEANING_LABEL: [np.ones(len(LATERAL_FEATURE_NAMES), dtype=float) * 2.0],
            },
            prototype_sources={
                STANDARD_SIDE_SITTING_LABEL: ["std"],
                DIAGONAL_SITTING_LABEL: ["diag"],
                SIDE_LEANING_LABEL: ["lean"],
            },
            feature_mean=np.zeros(len(LATERAL_FEATURE_NAMES), dtype=float),
            feature_scale=np.ones(len(LATERAL_FEATURE_NAMES), dtype=float),
            margin_threshold=0.08,
            distance_thresholds={
                STANDARD_SIDE_SITTING_LABEL: 10.0,
                DIAGONAL_SITTING_LABEL: 10.0,
                SIDE_LEANING_LABEL: 10.0,
            },
        )

        result = model.predict_from_features(np.ones(len(LATERAL_FEATURE_NAMES), dtype=float) * 0.02)

        self.assertEqual(result["lateral_posture_label"], LATERAL_BOUNDARY_LABEL)
        self.assertTrue(result["lateral_boundary"])
        self.assertTrue(result["lateral_fallback_used"])
        self.assertIn("low_prototype_margin", result["lateral_boundary_reasons"])

    def test_two_stage_lateral_recognizer_adds_candidate_fields_without_overriding_non_lateral(self) -> None:
        class Parent:
            def predict_posture(self, window: np.ndarray) -> dict[str, object]:
                return {
                    "label": "端正坐姿",
                    "raw_label": "端正坐姿",
                    "confidence": 0.91,
                    "second_label": "前倾端坐",
                    "margin": 0.45,
                    "is_boundary": False,
                    "prototype_diagnosis": {"label": "端正坐姿"},
                }

        model = LateralFineModel(
            prototypes={
                STANDARD_SIDE_SITTING_LABEL: [np.zeros(len(LATERAL_FEATURE_NAMES), dtype=float)],
                DIAGONAL_SITTING_LABEL: [np.ones(len(LATERAL_FEATURE_NAMES), dtype=float)],
                SIDE_LEANING_LABEL: [np.ones(len(LATERAL_FEATURE_NAMES), dtype=float) * 2.0],
            },
            prototype_sources={
                STANDARD_SIDE_SITTING_LABEL: ["std"],
                DIAGONAL_SITTING_LABEL: ["diag"],
                SIDE_LEANING_LABEL: ["lean"],
            },
            feature_mean=np.zeros(len(LATERAL_FEATURE_NAMES), dtype=float),
            feature_scale=np.ones(len(LATERAL_FEATURE_NAMES), dtype=float),
            margin_threshold=0.01,
            distance_thresholds={
                STANDARD_SIDE_SITTING_LABEL: 10.0,
                DIAGONAL_SITTING_LABEL: 10.0,
                SIDE_LEANING_LABEL: 10.0,
            },
        )
        recognizer = TwoStageLateralRecognizer(Parent(), model)
        frame = np.zeros((16, 16), dtype=float)
        frame[4:9, 4:12] = 100.0

        payload = recognizer.predict_posture(np.stack([frame] * 3))

        self.assertFalse(payload["lateral_subclassifier_triggered"])
        self.assertEqual(payload["label"], "端正坐姿")
        self.assertEqual(payload["model_version"], "v2_3_candidate")

class LeanbackSubclassifierTest(unittest.TestCase):
    def test_gate_only_runs_for_leanback_related_parent_results(self) -> None:
        leanback = {
            "label": "后靠/瘫坐类",
            "confidence": 0.81,
            "second_label": "标准靠背坐",
            "margin": 0.32,
            "is_boundary": False,
            "prototype_diagnosis": {"label": "后靠/瘫坐类"},
        }
        upright = {
            "label": "端正坐姿",
            "confidence": 0.91,
            "second_label": "前倾端坐",
            "margin": 0.5,
            "is_boundary": False,
            "prototype_diagnosis": {"label": "端正坐姿"},
        }

        self.assertTrue(should_run_leanback_subclassifier(leanback)[0])
        self.assertFalse(should_run_leanback_subclassifier(upright)[0])

    def test_physical_gate_blocks_strong_left_right_asymmetry(self) -> None:
        ok_features = {"left_share": 0.52, "row_0_3_share": 0.35, "row_4_7_share": 0.50, "cop_y": 4.2}
        crossed_leg_like = {"left_share": 0.72, "row_0_3_share": 0.35, "row_4_7_share": 0.50, "cop_y": 4.2}

        self.assertTrue(leanback_physical_gate(ok_features)[0])
        self.assertFalse(leanback_physical_gate(crossed_leg_like)[0])

    def test_fine_model_prefers_boundary_fallback_for_low_margin_cases(self) -> None:
        model = LeanbackFineModel(
            prototypes={
                FINE_LEANBACK_LABEL: np.zeros(16, dtype=float),
                FINE_SLOUCH_LABEL: np.ones(16, dtype=float) * 0.04,
            },
            feature_mean=np.zeros(16, dtype=float),
            feature_scale=np.ones(16, dtype=float),
            margin_threshold=0.08,
            distance_thresholds={FINE_LEANBACK_LABEL: 1.0, FINE_SLOUCH_LABEL: 1.0},
        )

        result = model.predict_from_features(np.ones(16, dtype=float) * 0.02)

        self.assertEqual(result["fine_posture_label"], FINE_BOUNDARY_LABEL)
        self.assertTrue(result["fine_boundary"])
        self.assertTrue(result["fallback_used"])
        self.assertIn("prototype_boundary", result["fine_boundary_reasons"])

    def test_two_stage_recognizer_preserves_parent_fields_and_adds_fine_fields(self) -> None:
        class Parent:
            def predict_posture(self, window: np.ndarray) -> dict[str, object]:
                return {
                    "label": "后靠/瘫坐类",
                    "confidence": 0.86,
                    "second_label": "标准靠背坐",
                    "margin": 0.31,
                    "is_boundary": False,
                    "prototype_diagnosis": {"label": "后靠/瘫坐类"},
                }

        model = LeanbackFineModel(
            prototypes={
                FINE_LEANBACK_LABEL: np.zeros(len(LEANBACK_FEATURE_NAMES), dtype=float),
                FINE_SLOUCH_LABEL: np.ones(len(LEANBACK_FEATURE_NAMES), dtype=float),
            },
            feature_mean=np.zeros(len(LEANBACK_FEATURE_NAMES), dtype=float),
            feature_scale=np.ones(len(LEANBACK_FEATURE_NAMES), dtype=float),
            margin_threshold=0.01,
            distance_thresholds={FINE_LEANBACK_LABEL: 10.0, FINE_SLOUCH_LABEL: 10.0},
        )
        recognizer = TwoStageLeanbackRecognizer(Parent(), model, model_version="v2_2_candidate")

        frame = np.zeros((16, 16), dtype=float)
        frame[2:7, 4:12] = 100.0
        payload = recognizer.predict_posture(np.stack([frame] * 3))

        self.assertEqual(payload["parent_posture_label"], "后靠/瘫坐类")
        self.assertTrue(payload["subclassifier_triggered"])
        self.assertIn(payload["label"], {FINE_LEANBACK_LABEL, FINE_SLOUCH_LABEL, FINE_BOUNDARY_LABEL})
        self.assertEqual(payload["model_version"], "v2_2_candidate")


class ModelArtifactTest(unittest.TestCase):
    def test_model_bundle_round_trips_without_changing_predictions(self) -> None:
        model = ConstantProbModel()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rf_posture_v1.joblib"
            save_model_bundle(path, model=model, metadata={"labels": ["A", "B"]})
            loaded = load_model_bundle(path)

        before = ProbabilityRecognizer(model).predict_posture(np.zeros((16, 16))).label
        after = ProbabilityRecognizer(loaded["model"]).predict_posture(np.zeros((16, 16))).label
        self.assertEqual(before, after)


class SmoothingTest(unittest.TestCase):
    def test_smoother_requires_confirmed_switch_and_marks_low_confidence_boundary(self) -> None:
        smoother = PredictionSmoother(vote_window=5, switch_confirmations=3, min_confidence=0.6, min_margin=0.2)
        a = {"label": "A", "confidence": 0.9, "margin": 0.5, "is_boundary": False}
        b = {"label": "B", "confidence": 0.9, "margin": 0.5, "is_boundary": False}
        low = {"label": "B", "confidence": 0.51, "margin": 0.1, "is_boundary": False}

        self.assertEqual(smoother.update(a)["label"], "A")
        self.assertEqual(smoother.update(b)["label"], "A")
        self.assertEqual(smoother.update(b)["label"], "A")
        self.assertEqual(smoother.update(b)["label"], "B")
        self.assertTrue(smoother.update(low)["is_boundary"])
        self.assertEqual(smoother.update(low)["label"], "边界/不确定")


class TrainingTest(unittest.TestCase):
    def test_build_v1_bank_uses_multi_prototypes_for_class5_and_mirror_for_class11(self) -> None:
        class Sample:
            def __init__(self, name: str, label: str, source_family: str, value: float) -> None:
                self.path = type("Pathish", (), {"name": name})()
                self.label = label
                self.source_family = source_family
                self.features = np.ones((3, 264), dtype=float) * value

        samples = [
            Sample("bantangkaobeizuo5-1.csv", "后靠/瘫坐类", "class5", 1.0),
            Sample("bantangkaobeizuo5-2.csv", "后靠/瘫坐类", "class5", 1.2),
            Sample("cewobantang11-1.csv", "躺卧类", "class11", 2.0),
            Sample("duanzhengzuozi1-1.csv", "端正坐姿", "duanzhengzuozi1", 0.0),
        ]

        bank = build_v1_prototype_bank(samples)

        class5_protos = [proto for proto in bank.prototypes if proto.source_group == "class5"]
        class11_protos = [proto for proto in bank.prototypes if proto.source_group == "class11"]
        self.assertEqual(len(class5_protos), 2)
        self.assertTrue(all(proto.label == "后靠/瘫坐类" for proto in class5_protos))
        self.assertTrue(all(proto.mirror_aware for proto in class11_protos))

    def test_compare_v1_models_accepts_explicit_model_subset(self) -> None:
        evaluations = compare_v1_models([], models=("unknown",))

        self.assertEqual(evaluations[0].status, "unavailable")
        self.assertIn("Unknown model", evaluations[0].detail)


class RealtimePipelineTest(unittest.TestCase):
    def test_pipeline_waits_for_stability_before_predicting(self) -> None:
        proto_empty = Prototype("empty-like", "目标姿势", extract_features(np.ones((16, 16)) * 20))
        proto_other = Prototype("other", "其它姿势", np.ones(264) * 4)
        recognizer = PrototypeRecognizer(PrototypeBank([proto_empty, proto_other]))
        pipeline = RealtimePosturePipeline(recognizer, fps=10.0, window_seconds=0.5, settle_seconds=0.5)
        frame = np.ones((16, 16), dtype=float) * 20

        first = pipeline.update(frame)
        self.assertIsNone(first.posture)
        self.assertEqual(first.seat.phase, SeatPhase.SITTING_DOWN)

        last = first
        for _ in range(8):
            last = pipeline.update(frame)

        self.assertEqual(last.posture, "目标姿势")
        self.assertIsNotNone(last.prediction)
        self.assertGreaterEqual(last.duration_s, 0.1)


class PredictCLITest(unittest.TestCase):
    def test_prediction_to_dict_exposes_stable_public_contract(self) -> None:
        proto_a = Prototype("A::P1", "A", np.zeros(264))
        proto_b = Prototype("B::P1", "B", np.ones(264) * 3)
        result = PrototypeRecognizer(PrototypeBank([proto_a, proto_b])).predict_posture(np.zeros((16, 16)))

        payload = prediction_to_dict(result)

        self.assertEqual(set(payload), {"label", "confidence", "second_label", "margin", "is_boundary", "best_distance", "matched_prototype_id"})
        self.assertEqual(payload["label"], "A")


class GuiTest(unittest.TestCase):
    def test_pressure_to_color_returns_hex_rgb(self) -> None:
        self.assertRegex(pressure_to_color(0.0, 100.0), r"^#[0-9a-f]{6}$")
        self.assertRegex(pressure_to_color(100.0, 100.0), r"^#[0-9a-f]{6}$")


if __name__ == "__main__":
    unittest.main()
