from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"


def load_tool(module_name: str):
    path = TOOLS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RuntimeBundleExportTest(unittest.TestCase):
    def test_runtime_bundle_exports_manifest_and_required_files(self) -> None:
        export_bundle = load_tool("export_runtime_bundle")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "v2_4_3_candidate"
            manifest = export_bundle.export_runtime_bundle(
                model_version="v2_4_3_candidate",
                output_dir=output_dir,
            )
            reloaded = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema_version"], "posture_runtime_bundle_v1")
            self.assertEqual(reloaded["model_version"], "v2_4_3_candidate")
            self.assertEqual(
                sorted(reloaded["files"]),
                [
                    "lateral_v243_classifier",
                    "leanback_classifier",
                    "main_classifier",
                    "parent_hybrid",
                    "runtime_config",
                ],
            )
            for key, info in reloaded["files"].items():
                path = output_dir / info["filename"]
                self.assertTrue(path.exists(), key)
                self.assertEqual(info["size"], path.stat().st_size)
                self.assertEqual(info["sha256"], sha256_file(path))
                self.assertIn("source_sha256", info)

    def test_runtime_bundle_does_not_use_default_model_pointer(self) -> None:
        export_bundle = load_tool("export_runtime_bundle")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "bundle"
            manifest = export_bundle.export_runtime_bundle(
                model_version="v2_4_3_candidate",
                output_dir=output_dir,
            )

            self.assertEqual(manifest["model_version"], "v2_4_3_candidate")
            self.assertEqual(
                json.loads((output_dir / "runtime_config.json").read_text(encoding="utf-8"))["model_version"],
                "v2_4_3_candidate",
            )


if __name__ == "__main__":
    unittest.main()
