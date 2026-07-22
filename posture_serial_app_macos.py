from __future__ import annotations

from typing import Sequence

from recognizer.serial_gui import main as serial_main


def main(argv: Sequence[str] | None = None) -> int:
    return serial_main(
        argv,
        brand_name="绿联智能",
        subtitle="实时串口坐姿识别系统",
        app_title="绿联智能｜实时串口坐姿识别系统",
    )


if __name__ == "__main__":
    raise SystemExit(main())
