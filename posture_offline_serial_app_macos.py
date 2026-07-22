from __future__ import annotations

from typing import Sequence

from recognizer.offline_analysis_gui import main as offline_main


def main(argv: Sequence[str] | None = None) -> int:
    return offline_main(
        argv,
        brand_name="绿联智能",
        subtitle="离线串口坐姿分析软件",
        app_title="绿联智能｜离线串口坐姿分析软件",
        model_version="v2_4_3_candidate",
    )


if __name__ == "__main__":
    raise SystemExit(main())
