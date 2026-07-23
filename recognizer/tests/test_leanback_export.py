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
        lines.append(f"2026/07/23_11:00:{index:02d}")
        for row in frame:
            lines.append(",".join(str(int(value)) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class LeanbackExportTest(unittest.TestCase):
    def test_leanback_model_export_matches_runtime_object(self) -> None:
        export_leanback_classifier = load_tool("export_leanback_classifier")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "leanback_classifier.json"
            payload = export_leanback_classifier.export_leanback_classifier("v2_4_3_candidate", target)
            reloaded = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "leanback_classifier_export_v1")
        self.assertEqual(reloaded["model_version"], "v2_4_3_candidate")
        self.assertEqual(reloaded["submodel_version"], "leanback_subclassifier_v2_2_candidate")
        self.assertEqual(reloaded["object_type"], "recognizer.leanback_subclassifier.LeanbackFineModel")
        self.assertEqual(reloaded["classes"], ["后仰靠背坐", "后靠/瘫坐类"])
        self.assertEqual(reloaded["fallback_label"], "后靠坐姿")
        self.assertEqual(reloaded["feature_dim"], 22)
        self.assertEqual(len(reloaded["feature_names"]), 22)
        self.assertEqual(len(reloaded["feature_mean"]), 22)
        self.assertEqual(len(reloaded["feature_scale"]), 22)
        self.assertEqual(set(reloaded["prototypes"]), {"后仰靠背坐", "后靠/瘫坐类"})
        self.assertIsNone(reloaded["classifier"])
        self.assertEqual(reloaded["margin_threshold"], 0.08)
        self.assertEqual(reloaded["confidence_threshold"], 0.55)
        self.assertEqual(reloaded["gate"]["parent_related_labels"], ["后仰靠背坐", "后靠/瘫坐类"])
        self.assertIn("source_joblib_sha256", reloaded)

    def test_leanback_golden_exports_trigger_and_passthrough_frames(self) -> None:
        export_leanback_golden = load_tool("export_leanback_golden")

        frame = np.zeros((16, 16), dtype=np.uint8)
        frame[:8, 6:10] = 30
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pressure_frames.csv"
            target = Path(tmp) / "leanback_golden.jsonl"
            write_pressure_csv(source, [frame])
            count = export_leanback_golden.export_leanback_golden(
                source,
                target,
                model_version="v2_4_3_candidate",
            )
            records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        record = records[0]
        self.assertEqual(record["schema_version"], "leanback_golden_v1")
        self.assertEqual(record["frame_index"], 0)
        self.assertEqual(len(record["frame_uint8"]), 256)
        self.assertEqual(record["frame_uint8"][0], int(frame.reshape(-1)[0]))
        self.assertIn("parent_hybrid", record)
        self.assertIn("leanback_triggered", record)
        self.assertIn("leanback_trigger_reasons", record)
        self.assertEqual(len(record["leanback_features"]), 22)
        self.assertEqual(len(record["leanback_feature_names"]), 22)
        self.assertEqual(record["leanback_classes"], ["后仰靠背坐", "后靠/瘫坐类"])
        self.assertIn("leanback_output_label", record)
        self.assertIn("leanback_output_confidence", record)
        self.assertIn("should_enter_lateral_stage", record)
        self.assertNotIn("lateral_posture_label", record)
        self.assertNotIn("final_display_label_after_lateral", record)


if __name__ == "__main__":
    unittest.main()
