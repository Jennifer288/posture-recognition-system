from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_main_classifier import export_main_classifier
from export_main_classifier_golden import export_main_classifier_golden


def write_sensor_csv(path: Path, frames: list[np.ndarray]) -> None:
    lines: list[str] = []
    for index, frame in enumerate(frames):
        lines.append(f"2026/07/23_10:30:{index:02d}")
        for row in np.asarray(frame, dtype=int):
            lines.append(",".join(str(int(value)) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class MainClassifierExportTest(unittest.TestCase):
    def test_export_model_contains_calibrated_random_forest_structure(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "main_classifier.json"

            payload = export_main_classifier("v2_4_3_candidate", output)

            self.assertEqual(payload["schema_version"], "main_classifier_export_v1")
            self.assertEqual(payload["model_version"], "v2_4_3_candidate")
            self.assertEqual(payload["classifier_type"], "sklearn.calibration.CalibratedClassifierCV")
            self.assertEqual(payload["calibration_method"], "sigmoid")
            self.assertEqual(payload["feature_dim"], 264)
            self.assertEqual(len(payload["classes"]), 14)
            self.assertEqual(len(payload["calibrated_classifiers"]), 3)
            first_fold = payload["calibrated_classifiers"][0]
            self.assertEqual(first_fold["estimator_type"], "sklearn.ensemble._forest.RandomForestClassifier")
            self.assertEqual(first_fold["tree_count"], 80)
            self.assertEqual(len(first_fold["calibrators"]), 14)
            first_tree = first_fold["trees"][0]
            self.assertIn("children_left", first_tree)
            self.assertIn("children_right", first_tree)
            self.assertIn("feature", first_tree)
            self.assertIn("threshold", first_tree)
            self.assertIn("value", first_tree)
            self.assertEqual(len(first_tree["children_left"]), first_tree["node_count"])
            self.assertEqual(len(first_tree["value"][0]), 14)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["source_joblib_sha256"], payload["source_joblib_sha256"])

    def test_export_golden_contains_main_classifier_outputs_not_final_resolver_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "pressure_frames.csv"
            output = tmp / "main_classifier_golden.jsonl"
            frame = np.zeros((16, 16), dtype=np.uint8)
            frame[5:10, 5:11] = 50
            write_sensor_csv(source, [np.zeros((16, 16), dtype=np.uint8), frame])

            count = export_main_classifier_golden(
                source,
                output,
                model_version="v2_4_3_candidate",
            )

            self.assertEqual(count, 2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            record = records[-1]
            self.assertEqual(record["schema_version"], "main_classifier_golden_v1")
            self.assertEqual(len(record["features"]), 264)
            self.assertEqual(record["classes"], records[0]["classes"])
            self.assertEqual(len(record["final_predict_proba"]), 14)
            self.assertAlmostEqual(sum(record["final_predict_proba"]), 1.0, places=12)
            self.assertIn("main_classifier_prediction", record)
            self.assertIn("main_classifier_confidence", record)
            self.assertIn("folds", record)
            self.assertEqual(len(record["folds"]), 3)
            self.assertNotIn("final_display_label", record)


if __name__ == "__main__":
    unittest.main()
