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
        lines.append(f"2026/07/23_10:00:{index:02d}")
        for row in frame:
            lines.append(",".join(str(int(value)) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ParentHybridExportTest(unittest.TestCase):
    def test_parent_hybrid_model_export_contains_runtime_rules_without_main_forest(self) -> None:
        export_parent_hybrid_model = load_tool("export_parent_hybrid_model")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "parent_hybrid.json"
            payload = export_parent_hybrid_model.export_parent_hybrid_model("v2_4_3_candidate", target)
            reloaded = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "parent_hybrid_model_v1")
        self.assertEqual(reloaded["model_version"], "v2_4_3_candidate")
        self.assertEqual(reloaded["feature_dim"], 264)
        self.assertEqual(reloaded["prototype_bank"]["version"], "recognizer_v1_prototype_bank")
        self.assertEqual(len(reloaded["prototype_bank"]["feature_mean"]), 264)
        self.assertEqual(len(reloaded["prototype_bank"]["feature_std"]), 264)
        self.assertEqual(len(reloaded["prototype_bank"]["prototypes"]), 14)
        self.assertEqual(reloaded["prototype_distance"]["metric"], "euclidean_l2")
        self.assertEqual(reloaded["prototype_distance"]["space"], "standardized_264_feature")
        self.assertEqual(reloaded["prototype_decision"]["confidence_round_decimals"], 4)
        self.assertEqual(reloaded["rf_boundary"]["min_confidence"], 0.55)
        self.assertEqual(reloaded["rf_boundary"]["boundary_margin"], 0.10)
        self.assertEqual(reloaded["hybrid_boundary"]["prototype_conflict_margin"], 0.18)
        self.assertEqual(reloaded["hybrid_boundary"]["prototype_boundary_confidence_gate"], 0.65)
        self.assertIn("prototype_bank_sha256", reloaded["source_files"])
        self.assertNotIn("calibrated_classifiers", reloaded)

    def test_parent_hybrid_golden_exports_frame_level_decisions(self) -> None:
        export_parent_hybrid_golden = load_tool("export_parent_hybrid_golden")

        frame = np.zeros((16, 16), dtype=np.uint8)
        frame[8, 7] = 120
        frame[8, 8] = 130
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pressure_frames.csv"
            target = Path(tmp) / "golden.jsonl"
            write_pressure_csv(source, [frame])
            count = export_parent_hybrid_golden.export_parent_hybrid_golden(
                source,
                target,
                model_version="v2_4_3_candidate",
            )
            records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        record = records[0]
        self.assertEqual(record["schema_version"], "parent_hybrid_golden_v1")
        self.assertEqual(record["feature_dim"], 264)
        self.assertEqual(len(record["main_classifier_predict_proba"]), 14)
        self.assertEqual(record["main_classifier_label"], record["boundary_before_label"])
        self.assertIn("prototype", record)
        self.assertTrue(record["prototype"]["entered"])
        self.assertEqual(len(record["prototype"]["class_distances"]), 14)
        self.assertIn("best_label", record["prototype"])
        self.assertIn("is_boundary", record["prototype"])
        self.assertIn("boundary_reasons", record)
        self.assertIn("parent_hybrid_label", record)
        self.assertIn("parent_hybrid_confidence", record)
        self.assertFalse(record["requires_leanback_subclassifier"])
        self.assertFalse(record["requires_lateral_subclassifier"])
        self.assertNotIn("final_display_label", record)


if __name__ == "__main__":
    unittest.main()
