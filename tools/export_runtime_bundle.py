#!/usr/bin/env python3
"""Export the complete C++ runtime model bundle for the LVGL simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_lateral_v243_classifier import export_lateral_v243_classifier
from export_leanback_classifier import export_leanback_classifier
from export_main_classifier import export_main_classifier
from export_parent_hybrid_model import export_parent_hybrid_model
from export_runtime_config import export_runtime_config


SCHEMA_VERSION = "posture_runtime_bundle_v1"
DEFAULT_MODEL_VERSION = "v2_4_3_candidate"


ExportFn = Callable[..., dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _source_sha256(payload: dict[str, Any]) -> str | None:
    for key in ("source_joblib_sha256", "runtime_config_sha256"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    source_files = payload.get("source_files")
    if isinstance(source_files, dict):
        pieces = [
            str(value)
            for key, value in sorted(source_files.items())
            if key.endswith("_sha256") and value
        ]
        if pieces:
            digest = hashlib.sha256()
            digest.update("\n".join(pieces).encode("utf-8"))
            return digest.hexdigest()
    pieces = [
        str(value)
        for key, value in sorted(payload.items())
        if key.startswith("source_") and key.endswith("_sha256") and value
    ]
    if pieces:
        digest = hashlib.sha256()
        digest.update("\n".join(pieces).encode("utf-8"))
        return digest.hexdigest()
    return None


def _source_paths(payload: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key, value in payload.items():
        if key.startswith("source_") and not key.endswith("_sha256") and isinstance(value, str):
            paths.append(value)
    source_files = payload.get("source_files")
    if isinstance(source_files, dict):
        for key, value in source_files.items():
            if not key.endswith("_sha256") and isinstance(value, str):
                paths.append(value)
    return sorted(dict.fromkeys(paths))


def _write_export(
    *,
    key: str,
    filename: str,
    output_dir: Path,
    model_version: str,
    export_fn: ExportFn,
) -> dict[str, Any]:
    target = output_dir / filename
    payload = export_fn(model_version=model_version, output_path=target)
    return {
        "filename": filename,
        "sha256": sha256_file(target),
        "size": target.stat().st_size,
        "schema_version": payload.get("schema_version"),
        "model_version": payload.get("model_version"),
        "source_paths": _source_paths(payload),
        "source_sha256": _source_sha256(payload),
        "bundle_key": key,
    }


def export_runtime_bundle(
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    output_dir: str | Path,
) -> dict[str, Any]:
    if model_version != DEFAULT_MODEL_VERSION:
        raise ValueError(f"this runtime bundle exporter currently requires {DEFAULT_MODEL_VERSION}")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, str, ExportFn]] = [
        ("main_classifier", "main_classifier.json", export_main_classifier),
        ("parent_hybrid", "parent_hybrid.json", export_parent_hybrid_model),
        ("leanback_classifier", "leanback_classifier.json", export_leanback_classifier),
        ("lateral_v243_classifier", "lateral_v243_classifier.json", export_lateral_v243_classifier),
        ("runtime_config", "runtime_config.json", export_runtime_config),
    ]

    files = {
        key: _write_export(
            key=key,
            filename=filename,
            output_dir=target_dir,
            model_version=model_version,
            export_fn=export_fn,
        )
        for key, filename, export_fn in specs
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": _dependency_version("numpy"),
        "sklearn_version": _dependency_version("sklearn"),
        "joblib_version": _dependency_version("joblib"),
        "files": files,
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_runtime_bundle(model_version=args.model_version, output_dir=args.output_dir)
    print(f"Exported runtime bundle {manifest['model_version']} to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
