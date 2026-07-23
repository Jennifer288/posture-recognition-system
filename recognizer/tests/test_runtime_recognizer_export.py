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


class RuntimeRecognizerExportTest(unittest.TestCase):
    def test_runtime_config_exports_real_stateful_defaults(self) -> None:
        export_runtime_config = load_tool("export_runtime_config")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "runtime_config.json"
            payload = export_runtime_config.export_runtime_config(
                model_version="v2_4_3_candidate",
                output_path=target,
            )
            reloaded = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "runtime_config_export_v1")
        self.assertEqual(reloaded["model_version"], "v2_4_3_candidate")
        self.assertEqual(reloaded["fps"], 20.0)
        self.assertEqual(reloaded["seat_analyzer"]["window_frames"], 30)
        self.assertEqual(reloaded["occupancy_detector"]["empty_total_threshold"], 80.0)
        self.assertEqual(reloaded["occupancy_detector"]["occupied_total_threshold"], 250.0)
        self.assertEqual(reloaded["seat_detector"]["settle_frames"], 20)
        self.assertEqual(reloaded["seat_detector"]["history_size"], 20)
        self.assertEqual(reloaded["smoother"]["vote_window"], 7)
        self.assertEqual(reloaded["smoother"]["switch_confirmations"], 3)
        self.assertIn("lateral_hold_frames", reloaded["lateral_stage"])
        self.assertIn("public_payload_keys", reloaded)

    def test_runtime_golden_uses_one_recognizer_and_exports_state(self) -> None:
        export_runtime_golden = load_tool("export_runtime_recognizer_golden")

        empty = np.zeros((16, 16), dtype=np.uint8)
        occupied = np.zeros((16, 16), dtype=np.uint8)
        occupied[4:12, 4:12] = 80
        frames = [empty, occupied, occupied]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pressure_frames.csv"
            target = Path(tmp) / "runtime_golden.jsonl"
            write_pressure_csv(source, frames)
            count = export_runtime_golden.export_runtime_recognizer_golden(
                source,
                target,
                model_version="v2_4_3_candidate",
            )
            records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        self.assertEqual(records[0]["schema_version"], "runtime_recognizer_golden_v1")
        self.assertEqual(records[0]["public_payload"]["occupancy"], "EMPTY")
        self.assertEqual(records[1]["frame_index"], 1)
        self.assertEqual(len(records[1]["frame_uint8"]), 256)
        self.assertIn("analyzer_state", records[1])
        self.assertIn("occupancy_debug", records[1])
        self.assertIn("seat_debug", records[1])
        self.assertIn("smoother_debug", records[1])
        self.assertIn("classification_executed", records[1])
        self.assertIn("public_payload", records[1])
        self.assertEqual(records[1]["analyzer_state"]["window_frames"], 30)
        self.assertGreaterEqual(records[2]["analyzer_state"]["recent_frame_count"], records[1]["analyzer_state"]["recent_frame_count"])


if __name__ == "__main__":
    unittest.main()
