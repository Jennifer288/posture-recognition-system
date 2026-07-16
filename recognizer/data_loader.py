from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .feature_extractor import extract_batch_features, windowed_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = PROJECT_ROOT / "posture_sensor_analysis" / "quality_batches" / "postures_4_7_acceptance"
DEFAULT_FORMAL_DATASET = QUALITY_ROOT / "dataset_v1_1_17_final"
DEFAULT_STAGE8_10_11 = QUALITY_ROOT / "stage8_10_11_raw"
DEFAULT_CLASS11_REPLACEMENT = QUALITY_ROOT / "class11_round2_replacement_raw"


ORIGINAL_LABELS = {
    "duanzhengzuozi1": "端正坐姿",
    "qianqingduanzuo2": "前倾端坐",
    "qianduanzuozi2": "前倾端坐",
    "duanzuoqianqing3": "端坐前倾探身",
    "duanzuoqianqingtanshen3": "端坐前倾探身",
    "biaozhunkaobeizuo4": "标准靠背坐",
    "bantangkaobeizuo5": "半躺靠背坐",
    "houyangkaobeizuo6": "后仰靠背坐",
    "jiaochatuikaobei7": "交叉腿靠背坐",
    "jiaochatuikaobeizuo7": "交叉腿靠背坐",
    "tanzuo8.9": "瘫坐/斜躺合并",
    "quantangwozi10": "全躺卧姿",
    "cewobantang11": "侧卧半躺",
    "pantuizuo12": "盘腿坐",
    "guizuo13": "跪坐",
    "quansuozuo14": "蜷缩坐",
    "biaozhuncezuo15": "标准侧坐",
    "xiekuazuo16": "斜跨坐",
    "ceshenyikaozuo17": "侧身倚靠坐",
    "xieshenyikaozuo17": "侧身倚靠坐",
}


def map_v1_label(original_label: str) -> str:
    if original_label in {"半躺靠背坐", "瘫坐/斜躺合并"}:
        return "后靠/瘫坐类"
    if original_label in {"全躺卧姿", "侧卧半躺"}:
        return "躺卧类"
    return original_label


def base_stem(path_or_name: str | Path) -> str:
    stem = Path(path_or_name).stem
    match = re.search(r"-\d+$", stem)
    return stem[: match.start()] if match else stem


def original_label_from_name(path_or_name: str | Path) -> str:
    base = base_stem(path_or_name)
    return ORIGINAL_LABELS.get(base, base)


def source_family_from_name(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    if name.startswith("bantangkaobeizuo5"):
        return "class5"
    if name.startswith("tanzuo8.9"):
        return "class8"
    if name.startswith("quantangwozi10"):
        return "class10"
    if name.startswith("cewobantang11"):
        return "class11"
    return base_stem(name)


def is_timestamp(line: str) -> bool:
    return bool(re.match(r"^\d{4}/\d{2}/\d{2}_\d{2}:\d{2}:\d{2}$", line.strip()))


def read_sensor_csv(path: Path | str) -> tuple[list[str], np.ndarray]:
    source = Path(path)
    lines = source.read_text(encoding="ascii").splitlines()
    timestamps: list[str] = []
    frames: list[list[list[float]]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not is_timestamp(line):
            raise ValueError(f"{source.name}:{i + 1} expected timestamp, got {line!r}")
        timestamps.append(line)
        i += 1
        frame = []
        for _ in range(16):
            parts = [part.strip() for part in lines[i].split(",")]
            if len(parts) != 16:
                raise ValueError(f"{source.name}:{i + 1} expected 16 values, got {len(parts)}")
            frame.append([float(part) for part in parts])
            i += 1
        frames.append(frame)
    return timestamps, np.asarray(frames, dtype=float)


def stable_frames(frames: np.ndarray, fps: float = 20.0) -> np.ndarray:
    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (16, 16):
        raise ValueError(f"Expected frames shaped (n,16,16), got {arr.shape}")
    if len(arr) == 0:
        return arr
    totals = arr.sum(axis=(1, 2))
    p95 = float(np.percentile(totals, 95))
    occupied_threshold = max(250.0, p95 * 0.20)
    occupied = np.flatnonzero(totals >= occupied_threshold)
    if len(occupied) == 0:
        return arr[:0]
    first = int(occupied[0])
    last = int(occupied[-1])
    trim = max(1, int(round(0.30 * fps)))
    start = min(first + trim, last)
    end = max(last - trim + 1, start + 1)
    return arr[start:end]


@dataclass(frozen=True)
class WindowSample:
    path: Path
    original_label: str
    label: str
    source_family: str
    windows: np.ndarray
    features: np.ndarray


def load_window_sample(path: Path | str, window: int = 8, step: int = 2) -> WindowSample:
    source = Path(path)
    _, frames = read_sensor_csv(source)
    stable = stable_frames(frames)
    windows = windowed_frames(stable, window=window, step=step)
    features = extract_batch_features(windows)
    original = original_label_from_name(source)
    return WindowSample(
        path=source,
        original_label=original,
        label=map_v1_label(original),
        source_family=source_family_from_name(source),
        windows=windows,
        features=features,
    )


def default_training_files(include_stage8_10_11: bool = True) -> list[Path]:
    manifest = DEFAULT_FORMAL_DATASET / "final_file_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    files = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("file", "")
            if not name:
                continue
            path = DEFAULT_FORMAL_DATASET / name
            if path.exists() and path.suffix == ".csv":
                files.append(path)
    if include_stage8_10_11:
        files.extend(sorted(DEFAULT_STAGE8_10_11.glob("tanzuo8.9-*.csv")))
        files.extend(sorted(DEFAULT_STAGE8_10_11.glob("quantangwozi10-*.csv")))
        class11_1 = DEFAULT_STAGE8_10_11 / "cewobantang11-1.csv"
        class11_2 = DEFAULT_CLASS11_REPLACEMENT / "cewobantang11-2.csv"
        class11_3 = DEFAULT_CLASS11_REPLACEMENT / "cewobantang11-3.csv"
        for path in [class11_1, class11_2, class11_3]:
            if path.exists():
                files.append(path)
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted(files, key=lambda item: item.name)
