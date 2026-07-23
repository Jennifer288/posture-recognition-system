from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from recognizer.feature_extractor import FEATURE_DIM, FEATURE_NAMES

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_recognizer_golden import export_golden


def write_sensor_csv(path: Path, frames: list[np.ndarray]) -> None:
    lines: list[str] = []
    for index, frame in enumerate(frames):
        lines.append(f"2026/07/23_10:29:{index:02d}")
        for row in np.asarray(frame, dtype=int):
            lines.append(",".join(str(int(value)) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class RecognizerGoldenExportTest(unittest.TestCase):
    def test_feature_names_match_static_feature_dimension(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertEqual(FEATURE_NAMES[0], "norm_r00_c00")
        self.assertEqual(FEATURE_NAMES[15], "norm_r00_c15")
        self.assertEqual(FEATURE_NAMES[16], "norm_r01_c00")
        self.assertEqual(FEATURE_NAMES[-8:], (
            "log_total_div_10",
            "cop_x_div_15",
            "cop_y_div_15",
            "left_right_balance",
            "front_back_balance",
            "normalized_spread",
            "peak_share",
            "active_share_gt_20",
        ))

    def test_export_golden_jsonl_contains_static_features_and_final_prediction(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "pressure_frames.csv"
            output = tmp / "golden.jsonl"
            frame = np.zeros((16, 16), dtype=np.uint8)
            frame[7, 8] = 120
            write_sensor_csv(source, [np.zeros((16, 16), dtype=np.uint8), frame])

            count = export_golden(source, output, model_version="v2_4_3_candidate", fps=20.0)

            self.assertEqual(count, 2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["frame_index"], 0)
            self.assertEqual(records[0]["static_occupancy_state"], "EMPTY")
            self.assertEqual(records[1]["feature_dim"], FEATURE_DIM)
            self.assertEqual(len(records[1]["features"]), FEATURE_DIM)
            self.assertEqual(records[1]["feature_names"], list(FEATURE_NAMES))
            self.assertEqual(records[1]["algorithm_frame"][7][8], 120)
            self.assertIn("python_final_payload", records[1])


if __name__ == "__main__":
    unittest.main()
