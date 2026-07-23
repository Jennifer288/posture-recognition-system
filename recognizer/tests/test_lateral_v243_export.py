from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"


def load_tool(module_name: str):
    path = TOOLS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_pressure_csv(path: Path, frames: list[np.ndarray]) -> None:
    lines: list[str] = []
    for index, frame in enumerate(frames):
        lines.append(f"2026/07/23_12:00:{index:02d}")
        for row in frame:
            lines.append(",".join(str(int(value)) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class LateralV243ExportTest(unittest.TestCase):
    def test_lateral_model_export_matches_runtime_object(self) -> None:
        export_lateral = load_tool("export_lateral_v243_classifier")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "lateral_v243_classifier.json"
            payload = export_lateral.export_lateral_v243_classifier("v2_4_3_candidate", target)
            reloaded = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "lateral_v243_classifier_export_v1")
        self.assertEqual(reloaded["model_version"], "v2_4_3_candidate")
        self.assertEqual(reloaded["submodel_version"], "lateral_merged_subclassifier_v2_4_3_candidate")
        self.assertEqual(reloaded["object_type"], "recognizer.lateral_merged_subclassifier_v243.LateralMergedFineModelV243")
        self.assertIsNone(reloaded["classifier"])
        self.assertEqual(reloaded["classes"], ["侧向坐姿", "斜跨坐"])
        self.assertEqual(reloaded["fallback_label"], "侧向姿势")
        self.assertEqual(reloaded["feature_dim"], 42)
        self.assertEqual(len(reloaded["feature_names"]), 42)
        self.assertEqual(len(reloaded["feature_mean"]), 42)
        self.assertEqual(len(reloaded["feature_scale"]), 42)
        self.assertEqual({key: len(value) for key, value in reloaded["prototypes"].items()}, {"侧向坐姿": 10, "斜跨坐": 4})
        self.assertEqual(reloaded["confidence_threshold"], 0.2)
        self.assertEqual(reloaded["margin_thresholds"], {"侧向坐姿": 0.18, "斜跨坐": 0.08})
        self.assertIn("cross_leg_lateral_competition", reloaded["gate"])
        self.assertIn("source_joblib_sha256", reloaded)

    def test_lateral_golden_exports_stateful_stage_records(self) -> None:
        export_lateral_golden = load_tool("export_lateral_v243_golden")

        frame = np.zeros((16, 16), dtype=np.uint8)
        frame[4:13, :8] = 35
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pressure_frames.csv"
            target = Path(tmp) / "lateral_v243_golden.jsonl"
            write_pressure_csv(source, [frame, frame, frame])
            count = export_lateral_golden.export_lateral_v243_golden(
                source,
                target,
                model_version="v2_4_3_candidate",
            )
            records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        record = records[-1]
        self.assertEqual(record["schema_version"], "lateral_v243_golden_v1")
        self.assertEqual(len(record["frame_uint8"]), 256)
        self.assertIn("parent_hybrid", record)
        self.assertIn("leanback_stage", record)
        self.assertIn("lateral_triggered", record)
        self.assertIn("lateral_trigger_reasons", record)
        self.assertEqual(len(record["lateral_features"]), 42)
        self.assertEqual(len(record["lateral_feature_names"]), 42)
        self.assertEqual(record["lateral_classes"], ["侧向坐姿", "斜跨坐"])
        self.assertIn("lateral_output_label", record)
        self.assertIn("lateral_output_confidence", record)
        self.assertIn("lateral_output_raw_label", record)
        self.assertIn("smoothing_input", record)


if __name__ == "__main__":
    unittest.main()
