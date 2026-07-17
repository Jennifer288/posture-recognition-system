from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Sequence

import numpy as np

from .csv_gui_core import (
    CsvFormatError,
    CsvGuiError,
    CsvPlaybackData,
    CsvRecognitionSession,
    FramePrediction,
    load_csv_playback,
    load_runtime_recognizer,
)
from .gui import pressure_to_color
from .recognizer_api import default_model_version


STATE_LABELS = {
    "EMPTY": "空载",
    "LOAD_BELOW_THRESHOLD": "低于有效负载阈值",
    "OBJECT": "物品占用",
    "UNKNOWN": "未知占用",
    "HUMAN_STABILIZING": "人体正在稳定",
    "HUMAN_RECOGNIZING": "人体稳定识别中",
    "POSTURE": "姿势识别中",
}


class PostureCsvApp:
    def __init__(self, root: tk.Tk, model_version: str | None = None) -> None:
        self.root = root
        self.model_version = default_gui_model_version() if model_version is None else model_version
        self.root.title(f"CSV 坐姿识别软件 - 当前模型：{model_version_display_name(self.model_version)}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.data: CsvPlaybackData | None = None
        self.session: CsvRecognitionSession | None = None
        self.recognizer: object | None = None
        self.current_frame: np.ndarray = np.zeros((16, 16), dtype=float)
        self.current_record: FramePrediction | None = None
        self.after_id: str | None = None
        self.dragging = False
        self.cell_size = 28

        self.model_var = tk.StringVar(value=f"当前模型：{model_version_display_name(self.model_version)}")
        self.file_var = tk.StringVar(value="未选择CSV")
        self.state_var = tk.StringVar(value="空载")
        self.posture_var = tk.StringVar(value="—")
        self.raw_var = tk.StringVar(value="—")
        self.confidence_var = tk.StringVar(value="—")
        self.second_var = tk.StringVar(value="—")
        self.margin_var = tk.StringVar(value="—")
        self.boundary_var = tk.StringVar(value="否")
        self.boundary_reason_var = tk.StringVar(value="—")
        self.prototype_var = tk.StringVar(value="—")
        self.frame_var = tk.StringVar(value="0 / 0")
        self.time_var = tk.StringVar(value="0.00s")
        self.duration_var = tk.StringVar(value="0.00s")
        self.delay_var = tk.StringVar(value="—")
        self.total_var = tk.StringVar(value="0.0")
        self.max_var = tk.StringVar(value="0.0")
        self.active_var = tk.StringVar(value="0")
        self.speed_var = tk.StringVar(value="1×")
        self.fps_var = tk.StringVar(value="20 FPS")
        self.summary_var = tk.StringVar(value="请选择CSV文件开始。")

        self._build_ui()
        self._draw_heatmap(self.current_frame)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        for index in range(12):
            toolbar.columnconfigure(index, weight=0)

        ttk.Button(toolbar, text="选择CSV", command=self.select_csv).grid(row=0, column=0, padx=3)
        ttk.Button(toolbar, text="开始", command=self.start).grid(row=0, column=1, padx=3)
        ttk.Button(toolbar, text="暂停", command=self.pause).grid(row=0, column=2, padx=3)
        ttk.Button(toolbar, text="继续", command=self.resume).grid(row=0, column=3, padx=3)
        ttk.Button(toolbar, text="停止", command=self.stop).grid(row=0, column=4, padx=3)
        ttk.Button(toolbar, text="重新播放", command=self.replay).grid(row=0, column=5, padx=3)
        ttk.Button(toolbar, text="上一帧", command=self.previous_frame).grid(row=0, column=6, padx=3)
        ttk.Button(toolbar, text="下一帧", command=self.next_frame).grid(row=0, column=7, padx=3)
        ttk.Button(toolbar, text="重新校准", command=self.calibrate).grid(row=0, column=8, padx=3)
        ttk.Button(toolbar, text="导出结果", command=self.export_results).grid(row=0, column=9, padx=3)
        ttk.Label(toolbar, text="速度").grid(row=0, column=10, padx=(12, 3))
        ttk.Combobox(toolbar, textvariable=self.speed_var, values=["0.5×", "1×", "2×", "5×", "最大速度"], width=8, state="readonly").grid(row=0, column=11)
        ttk.Label(toolbar, text="采样率").grid(row=0, column=12, padx=(12, 3))
        ttk.Combobox(toolbar, textvariable=self.fps_var, values=["5 FPS", "10 FPS", "20 FPS"], width=8, state="readonly").grid(row=0, column=13)

        left = ttk.Frame(self.root, padding=(12, 8))
        left.grid(row=1, column=0, sticky="nsew")
        left.columnconfigure(1, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="前", anchor="center").grid(row=0, column=1, sticky="ew")
        ttk.Label(left, text="左", anchor="center").grid(row=1, column=0, sticky="ns")
        self.canvas = tk.Canvas(
            left,
            width=16 * self.cell_size,
            height=16 * self.cell_size,
            bg="#101828",
            highlightthickness=0,
        )
        self.canvas.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        ttk.Label(left, text="右", anchor="center").grid(row=1, column=2, sticky="ns")
        ttk.Label(left, text="后", anchor="center").grid(row=2, column=1, sticky="ew")

        stats = ttk.Frame(left)
        stats.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(stats, text="总压力").grid(row=0, column=0, padx=8)
        ttk.Label(stats, textvariable=self.total_var).grid(row=0, column=1, padx=8)
        ttk.Label(stats, text="最大压力").grid(row=0, column=2, padx=8)
        ttk.Label(stats, textvariable=self.max_var).grid(row=0, column=3, padx=8)
        ttk.Label(stats, text="活跃点").grid(row=0, column=4, padx=8)
        ttk.Label(stats, textvariable=self.active_var).grid(row=0, column=5, padx=8)

        right = ttk.Frame(self.root, padding=(8, 8, 12, 8))
        right.grid(row=1, column=1, sticky="nsew")
        self._build_status_panel(right)

        bottom = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.progress = ttk.Scale(bottom, from_=0, to=0, orient="horizontal", command=self._on_progress_drag)
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress.bind("<ButtonPress-1>", lambda _event: self._begin_drag())
        self.progress.bind("<ButtonRelease-1>", lambda _event: self._end_drag())
        ttk.Label(bottom, textvariable=self.summary_var).grid(row=1, column=0, sticky="w", pady=(5, 0))

    def _build_status_panel(self, parent: ttk.Frame) -> None:
        file_box = ttk.LabelFrame(parent, text="文件和回放", padding=8)
        file_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._field(file_box, "模型", self.model_var, 0)
        self._field(file_box, "文件", self.file_var, 1)
        self._field(file_box, "帧", self.frame_var, 2)
        self._field(file_box, "时间", self.time_var, 3)
        self._field(file_box, "持续", self.duration_var, 4)
        self._field(file_box, "识别延迟", self.delay_var, 5)

        state_box = ttk.LabelFrame(parent, text="当前系统状态", padding=8)
        state_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(state_box, textvariable=self.state_var, font=("Helvetica", 20, "bold")).grid(row=0, column=0, sticky="w")

        result_box = ttk.LabelFrame(parent, text="识别结果", padding=8)
        result_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._field(result_box, "姿势", self.posture_var, 0)
        self._field(result_box, "原始候选", self.raw_var, 1)
        self._field(result_box, "confidence", self.confidence_var, 2)
        self._field(result_box, "second_label", self.second_var, 3)
        self._field(result_box, "margin", self.margin_var, 4)
        self._field(result_box, "Boundary", self.boundary_var, 5)
        self._field(result_box, "原因", self.boundary_reason_var, 6)
        self._field(result_box, "Prototype", self.prototype_var, 7)

        history_box = ttk.LabelFrame(parent, text="识别历史", padding=8)
        history_box.grid(row=3, column=0, sticky="nsew")
        parent.rowconfigure(3, weight=1)
        columns = ("start", "end", "duration", "state", "posture", "mean_conf", "boundary")
        self.history = ttk.Treeview(history_box, columns=columns, show="headings", height=10)
        labels = {
            "start": "开始",
            "end": "结束",
            "duration": "时长",
            "state": "状态",
            "posture": "姿势",
            "mean_conf": "均值conf",
            "boundary": "Boundary",
        }
        widths = {"start": 60, "end": 60, "duration": 60, "state": 96, "posture": 110, "mean_conf": 70, "boundary": 70}
        for col in columns:
            self.history.heading(col, text=labels[col])
            self.history.column(col, width=widths[col], anchor="center")
        self.history.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(history_box, orient="vertical", command=self.history.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.history.configure(yscrollcommand=scrollbar.set)

    def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=variable, width=28).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)

    def select_csv(self) -> None:
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        path = filedialog.askopenfilename(
            title="选择FlexPressureVision CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            fps = float(self.fps_var.get().split()[0])
            self.data = load_csv_playback(Path(path), fallback_fps=fps)
            self.session = None
            self.current_record = None
            self.current_frame = self.data.frames[0]
            self._reset_recognizer_state()
            self.file_var.set(self.data.path.name)
            self.frame_var.set(f"0 / {self.data.frame_count}")
            self.time_var.set("0.00s")
            self.progress.configure(to=max(self.data.frame_count - 1, 0))
            self.progress.set(0)
            self.summary_var.set(f"已加载 {self.data.path.name}，{self.data.frame_count} 帧，时间来源：{self.data.time_source}")
            self._draw_heatmap(self.current_frame)
            self._clear_result_display()
        except CsvFormatError as exc:
            messagebox.showerror("CSV格式错误", str(exc))
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))

    def start(self) -> None:
        if self.data is None:
            messagebox.showwarning("未选择CSV", "请先点击“选择CSV”。")
            return
        try:
            self._ensure_session(reset=True)
        except CsvGuiError as exc:
            messagebox.showerror("模型加载失败", str(exc))
            return
        assert self.session is not None
        self.session.resume()
        self._schedule_next(immediate=True)

    def pause(self) -> None:
        if self.session:
            self.session.pause()
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def resume(self) -> None:
        if self.session is None:
            return
        self.session.resume()
        self._schedule_next(immediate=True)

    def stop(self) -> None:
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.session:
            self.session.stop()
        self.progress.set(0)
        self._clear_result_display()
        if self.data is not None:
            self._draw_heatmap(self.data.frames[0])

    def replay(self) -> None:
        if self.session:
            self.session.reset()
        self.start()

    def previous_frame(self) -> None:
        if self.data is None:
            return
        self._ensure_session(reset=False)
        assert self.session is not None
        target = max(0, self.session.index - 2)
        self.pause()
        record = self.session.seek(target)
        self._render_record(record)

    def next_frame(self) -> None:
        if self.data is None:
            return
        self._ensure_session(reset=False)
        assert self.session is not None
        self.pause()
        record = self.session.step_once()
        self._render_record(record)

    def calibrate(self) -> None:
        if self.recognizer is None:
            try:
                self.recognizer = load_runtime_recognizer(model_version=self.model_version)
            except CsvGuiError as exc:
                messagebox.showerror("模型加载失败", str(exc))
                return
        calibrate = getattr(self.recognizer, "calibrate", None)
        if callable(calibrate):
            calibrate(frame=self.current_frame)
            if self.session:
                self.session.reset()
            self.summary_var.set("已使用当前帧重新校准。请确认当前坐垫为空载。")

    def export_results(self) -> None:
        if self.session is None or not self.session.predictions:
            messagebox.showwarning("无结果", "请先开始识别，再导出结果。")
            return
        paths = self.session.export_results()
        messagebox.showinfo("导出完成", f"结果已导出到：\n{paths['directory']}")

    def _ensure_session(self, reset: bool) -> None:
        if self.data is None:
            raise CsvGuiError("未选择CSV")
        if self.recognizer is None:
            self.recognizer = load_runtime_recognizer(model_version=self.model_version)
        if self.session is None:
            self.session = CsvRecognitionSession(self.data, self.recognizer)
        elif reset:
            self.session.reset()
        self._refresh_history()

    def _schedule_next(self, immediate: bool = False) -> None:
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        delay = 1 if immediate else self._next_delay_ms()
        self.after_id = self.root.after(delay, self._tick)

    def _tick(self) -> None:
        self.after_id = None
        if self.session is None or not self.session.playing:
            return
        record = self.session.step()
        if record is None:
            self._finish_playback()
            return
        self._render_record(record)
        if self.session.index >= self.session.data.frame_count:
            self._finish_playback()
            return
        self._schedule_next()

    def _next_delay_ms(self) -> int:
        if self.session is None:
            return 50
        if self.speed_var.get() == "最大速度":
            return 1
        speed = float(self.speed_var.get().replace("×", ""))
        index = max(self.session.index - 1, 0)
        interval = self.session.data.frame_interval_s(index)
        return max(1, int(round(interval * 1000 / max(speed, 1e-6))))

    def _render_record(self, record: FramePrediction | None) -> None:
        if record is None or self.data is None:
            return
        self.current_record = record
        self.current_frame = self.data.frames[record.frame_index]
        self._draw_heatmap(self.current_frame)
        self.progress.set(record.frame_index)
        self.frame_var.set(f"{record.frame_index + 1} / {self.data.frame_count}")
        self.time_var.set(f"{record.timestamp:.2f}s")
        self.total_var.set(f"{record.total_pressure:.1f}")
        self.max_var.set(f"{record.max_pressure:.1f}")
        self.active_var.set(str(record.active_points))
        self.state_var.set(STATE_LABELS.get(record.display_status, record.display_status))
        self.posture_var.set(record.posture or ("边界姿势/低置信度" if record.display_status == "POSTURE" and record.is_boundary else "—"))
        self.raw_var.set(_format_raw_candidate(record))
        self.confidence_var.set(_format_optional(record.posture_confidence))
        self.second_var.set(record.second_label or "—")
        self.margin_var.set(_format_optional(record.margin))
        self.boundary_var.set("是" if record.is_boundary else "否")
        self.boundary_reason_var.set(record.boundary_reason or "—")
        self.prototype_var.set(self._prototype_label(record))
        self._refresh_history()
        summary = self.session.summary() if self.session else {}
        delay = summary.get("first_posture_delay_s")
        self.delay_var.set("—" if delay is None else f"{delay:.2f}s")
        self.duration_var.set(self._current_segment_duration())

    def _prototype_label(self, record: FramePrediction) -> str:
        if record.lateral_prototype_label:
            return f"lateral={record.lateral_prototype_label}; d={_format_optional(record.lateral_prototype_distance)}"
        if record.prototype_diagnosis:
            return record.prototype_diagnosis
        if record.raw_label or record.second_label:
            return f"second={record.second_label or '—'}"
        return "—"

    def _current_segment_duration(self) -> str:
        if not self.session or not self.session.segments:
            return "0.00s"
        return f"{self.session.segments[-1].duration:.2f}s"

    def _refresh_history(self) -> None:
        for item in self.history.get_children():
            self.history.delete(item)
        if not self.session:
            return
        for segment in self.session.segments[-50:]:
            self.history.insert(
                "",
                "end",
                values=(
                    f"{segment.start_time:.2f}",
                    f"{segment.end_time:.2f}",
                    f"{segment.duration:.2f}",
                    STATE_LABELS.get(segment.occupancy_state, segment.occupancy_state),
                    segment.posture or "—",
                    "—" if segment.mean_confidence is None else f"{segment.mean_confidence:.2f}",
                    f"{segment.boundary_ratio:.2f}",
                ),
            )

    def _finish_playback(self) -> None:
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if not self.session:
            return
        summary = self.session.summary()
        self.summary_var.set(
            "回放结束："
            f"主要姿势={summary.get('main_posture') or '—'}，"
            f"平均confidence={summary.get('average_confidence') or '—'}，"
            f"Boundary比例={summary.get('boundary_ratio')}"
        )
        messagebox.showinfo("识别总结", json.dumps(summary, ensure_ascii=False, indent=2))

    def _draw_heatmap(self, frame: np.ndarray) -> None:
        self.canvas.delete("all")
        max_value = float(np.max(frame)) if frame.size else 0.0
        for row in range(16):
            for col in range(16):
                x0 = col * self.cell_size
                y0 = row * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                value = float(frame[row, col])
                color = pressure_to_color(value, max_value)
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#172033")
                if value > 0:
                    self.canvas.create_text(
                        (x0 + x1) / 2,
                        (y0 + y1) / 2,
                        text=str(int(round(value))),
                        fill="#f8fafc" if value > max(max_value * 0.45, 1.0) else "#0f172a",
                        font=("Helvetica", 7),
                    )

    def _clear_result_display(self) -> None:
        self.current_record = None
        self.state_var.set("空载")
        self.posture_var.set("—")
        self.raw_var.set("—")
        self.confidence_var.set("—")
        self.second_var.set("—")
        self.margin_var.set("—")
        self.boundary_var.set("否")
        self.boundary_reason_var.set("—")
        self.prototype_var.set("—")
        self.duration_var.set("0.00s")
        self.delay_var.set("—")
        self.total_var.set("0.0")
        self.max_var.set("0.0")
        self.active_var.set("0")
        for item in self.history.get_children():
            self.history.delete(item)

    def _begin_drag(self) -> None:
        self.dragging = True
        self.pause()
        self.summary_var.set("拖动预览中：释放后会从CSV开头快速重放到目标帧并重建识别状态。")

    def _end_drag(self) -> None:
        self.dragging = False
        self._seek_to_progress()

    def _on_progress_drag(self, _value: str) -> None:
        if self.dragging:
            return

    def _seek_to_progress(self) -> None:
        if self.data is None:
            return
        self._ensure_session(reset=False)
        assert self.session is not None
        record = self.session.seek(int(round(float(self.progress.get()))))
        self._render_record(record)
        self.summary_var.set("已从CSV开头快速重放到目标帧，当前识别状态已重建。")

    def _on_close(self) -> None:
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.root.destroy()

    def _reset_recognizer_state(self) -> None:
        reset = getattr(self.recognizer, "reset", None)
        if callable(reset):
            reset()


def _format_optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _format_raw_candidate(record: FramePrediction) -> str:
    if record.raw_label:
        confidence = _format_optional(record.raw_confidence)
        return f"{record.raw_label} {confidence}"
    return "—"


def model_version_display_name(model_version: str) -> str:
    names = {
        "v1": "V1",
        "v2_candidate": "V2 Candidate",
        "v2_1_candidate": "V2.1（Phase 1闭卷通过）",
        "v2_2_candidate": "V2.2（H3闭卷通过）",
        "v2_3_candidate": "V2.3候选（侧向三类局部解析，未闭卷）",
        "v2_3_1_candidate": "V2.3.1候选（侧向链路修复，未闭卷）",
        "v2_4_candidate": "V2.4候选（侧向标签合并，未闭卷）",
        "v2_4_1_candidate": "V2.4.1候选（侧向合并边界修复，未闭卷）",
        "v2_4_2_candidate": "V2.4.2候选（斜跨门控修复，未闭卷）",
    }
    return names.get(model_version, model_version)


def default_gui_model_version() -> str:
    return default_model_version()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSV posture recognition desktop app.")
    parser.add_argument("--csv", help="Optional CSV file to preload.")
    parser.add_argument(
        "--model-version",
        default=default_gui_model_version(),
        choices=["v1", "v2_candidate", "v2_1_candidate", "v2_2_candidate", "v2_3_candidate", "v2_3_1_candidate", "v2_4_candidate", "v2_4_1_candidate", "v2_4_2_candidate"],
        help="Recognizer model version to load. Default follows recognizer/models/default_model.json.",
    )
    args = parser.parse_args(argv)

    root = tk.Tk()
    app = PostureCsvApp(root, model_version=args.model_version)
    if args.csv:
        try:
            app.data = load_csv_playback(Path(args.csv))
            app.current_frame = app.data.frames[0]
            app.file_var.set(app.data.path.name)
            app.progress.configure(to=max(app.data.frame_count - 1, 0))
            app._draw_heatmap(app.current_frame)
            app._clear_result_display()
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
