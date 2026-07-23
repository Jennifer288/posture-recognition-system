from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Sequence

import numpy as np

from .csv_gui import (
    STATE_LABELS,
    heatmap_grid_geometry,
    model_runtime_versions,
    model_version_display_name,
)
from .csv_gui_core import CsvGuiError, FramePrediction, load_runtime_recognizer, model_export_info
from .frame_reader import DEFAULT_SERIAL_BAUDRATE, SerialFrameReader, list_serial_ports
from .gui import pressure_to_color
from .serial_gui_core import ORIENTATION_MODES, RecognitionWorker, SerialRecognitionResult
from .serial_recorder import SERIAL_TEXT_FILENAME, SerialDataRecorder


DEFAULT_MODEL_VERSION = "v2_4_3_candidate"
DEFAULT_APP_TITLE_BASE = "实时串口坐姿识别"
DEFAULT_WINDOW_GEOMETRY = "1240x840"
MIN_WINDOW_SIZE = (1160, 780)
SERIAL_FONT_FAMILY = "TkDefaultFont"
SERIAL_FONT_SIZES = {
    "brand_title": 20,
    "brand_subtitle": 13,
    "section_title": 16,
    "field": 13,
    "field_value": 13,
    "summary_posture": 14,
    "summary_meta": 13,
    "button": 12,
    "input": 12,
    "direction": 14,
    "stats": 13,
    "helper": 11,
    "footer": 12,
}
SERIAL_SECTION_FRAME_STYLE = "SerialSection.TLabelframe"
SERIAL_INFO_LABEL_STYLE = "SerialInfo.TLabel"
SERIAL_INFO_VALUE_STYLE = "SerialInfoValue.TLabel"
SERIAL_WEAK_INFO_LABEL_STYLE = "SerialWeakInfo.TLabel"
SERIAL_SUMMARY_TITLE_STYLE = "SerialSummaryTitle.TLabel"
SERIAL_SUMMARY_POSTURE_STYLE = "SerialSummaryPosture.TLabel"
SERIAL_SUMMARY_META_STYLE = "SerialSummaryMeta.TLabel"
SERIAL_CONTROL_LABEL_STYLE = "SerialControl.TLabel"
SERIAL_DIRECTION_LABEL_STYLE = "SerialDirection.TLabel"
SERIAL_STATS_LABEL_STYLE = "SerialStats.TLabel"
SERIAL_HELP_LABEL_STYLE = "SerialHelp.TLabel"
SERIAL_FOOTER_LABEL_STYLE = "SerialFooter.TLabel"
SERIAL_PRIMARY_BUTTON_STYLE = "SerialPrimary.TButton"
SERIAL_DANGER_BUTTON_STYLE = "SerialDanger.TButton"
SERIAL_SECONDARY_BUTTON_STYLE = "SerialSecondary.TButton"
SERIAL_ENTRY_STYLE = "Serial.TEntry"
SERIAL_COMBOBOX_STYLE = "Serial.TCombobox"


def serial_window_title(model_version: str, *, app_title: str | None = None) -> str:
    if app_title:
        return app_title
    return f"{DEFAULT_APP_TITLE_BASE} - 当前模型：{model_version_display_name(model_version)}"


def serial_subtitle_text(model_version: str, subtitle: str) -> str:
    return f"{subtitle} · {model_version_display_name(model_version)}"


class PostureSerialApp:
    def __init__(
        self,
        root: tk.Tk,
        model_version: str = DEFAULT_MODEL_VERSION,
        *,
        app_title: str | None = None,
        brand_name: str | None = None,
        subtitle: str | None = None,
    ) -> None:
        self.root = root
        self.model_version = model_version
        self.app_title = app_title
        self.brand_name = brand_name
        self.subtitle = subtitle
        self.root.title(serial_window_title(self.model_version, app_title=self.app_title))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.geometry(DEFAULT_WINDOW_GEOMETRY)
        self.root.minsize(*MIN_WINDOW_SIZE)
        self._configure_styles()

        self.recognizer: object | None = None
        self.reader: SerialFrameReader | None = None
        self.worker: RecognitionWorker | None = None
        self.recorder: SerialDataRecorder | None = None
        self.result_queue: Queue[SerialRecognitionResult] | None = None
        self.connection_start_time: float | None = None
        self.current_frame: np.ndarray | None = None
        self.current_record: FramePrediction | None = None
        self.connecting = False
        self.poll_after_id: str | None = None
        self.model_details_window: tk.Toplevel | None = None
        self._reported_reader_error: str | None = None
        self._reported_worker_error: str | None = None

        self.port_var = tk.StringVar(value="")
        self.orientation_var = tk.StringVar(value=ORIENTATION_MODES[0])
        self.orientation_state = _ThreadSafeValue(self.orientation_var.get())
        self.connection_var = tk.StringVar(value="未连接")
        self.current_port_var = tk.StringVar(value="—")
        self.baud_var = tk.StringVar(value=str(DEFAULT_SERIAL_BAUDRATE))
        self.received_bytes_var = tk.StringVar(value="0")
        self.valid_frames_var = tk.StringVar(value="0")
        self.invalid_frames_var = tk.StringVar(value="0")
        self.discarded_bytes_var = tk.StringVar(value="0")
        self.receive_fps_var = tk.StringVar(value="0.0")
        self.frame_queue_var = tk.StringVar(value="0")
        self.dropped_queue_var = tk.StringVar(value="0")
        self.serial_error_var = tk.StringVar(value="—")
        self.summary_var = tk.StringVar(value="请选择串口并点击“连接”。")

        self.capture_label_var = tk.StringVar(value="")
        self.capture_trial_var = tk.StringVar(value="1")
        self.capture_dir_var = tk.StringVar(value="—")
        self.capture_status_var = tk.StringVar(value="未采集")
        self.capture_duration_var = tk.StringVar(value="0.00s")
        self.capture_raw_bytes_var = tk.StringVar(value="0")
        self.capture_frames_var = tk.StringVar(value="0")
        self.capture_predictions_var = tk.StringVar(value="0")
        self.capture_queue_var = tk.StringVar(value="0")
        self.capture_dropped_var = tk.StringVar(value="0")
        self.capture_error_var = tk.StringVar(value="—")
        self.capture_output_root: Path | None = None

        self.state_var = tk.StringVar(value="空载")
        self.summary_status_var = tk.StringVar(value="空载")
        self.summary_posture_var = tk.StringVar(value="—")
        self.summary_confidence_var = tk.StringVar(value="—")
        self.summary_boundary_var = tk.StringVar(value="否")
        self.occupancy_var = tk.StringVar(value="—")
        self.posture_var = tk.StringVar(value="—")
        self.raw_var = tk.StringVar(value="—")
        self.confidence_var = tk.StringVar(value="—")
        self.second_var = tk.StringVar(value="—")
        self.margin_var = tk.StringVar(value="—")
        self.boundary_var = tk.StringVar(value="否")
        self.boundary_reason_var = tk.StringVar(value="—")
        self.prototype_var = tk.StringVar(value="—")
        self.frame_index_var = tk.StringVar(value="0")
        self.uptime_var = tk.StringVar(value="0.00s")
        self.inference_var = tk.StringVar(value="—")
        self.average_inference_var = tk.StringVar(value="—")
        self.recognition_error_var = tk.StringVar(value="—")

        self.total_var = tk.StringVar(value="0.0")
        self.max_var = tk.StringVar(value="0.0")
        self.max_adc_var = tk.StringVar(value="当前最大ADC：0")
        self.active_var = tk.StringVar(value="0")
        self._wrap_labels: list[ttk.Label] = []

        self._build_ui()
        self._draw_heatmap(np.zeros((16, 16), dtype=np.float32))
        self.refresh_ports()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.configure(f"{SERIAL_SECTION_FRAME_STYLE}.Label", font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["section_title"], "bold"))
        style.configure(SERIAL_INFO_LABEL_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["field"]))
        style.configure(SERIAL_INFO_VALUE_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["field_value"]))
        style.configure(SERIAL_WEAK_INFO_LABEL_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["field"]))
        style.configure(SERIAL_SUMMARY_TITLE_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["summary_meta"]))
        style.configure(SERIAL_SUMMARY_POSTURE_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["summary_posture"], "bold"))
        style.configure(SERIAL_SUMMARY_META_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["summary_meta"]))
        style.configure(SERIAL_CONTROL_LABEL_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["field"]))
        style.configure(SERIAL_DIRECTION_LABEL_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["direction"], "bold"))
        style.configure(SERIAL_STATS_LABEL_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["stats"]))
        style.configure(SERIAL_HELP_LABEL_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["helper"]))
        style.configure(SERIAL_FOOTER_LABEL_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["footer"]))
        style.configure(SERIAL_PRIMARY_BUTTON_STYLE, padding=(8, 5), font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["button"], "bold"))
        style.configure(SERIAL_DANGER_BUTTON_STYLE, padding=(8, 5), font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["button"]))
        style.configure(SERIAL_SECONDARY_BUTTON_STYLE, padding=(6, 5), font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["button"]))
        style.configure(SERIAL_ENTRY_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["input"]))
        style.configure(SERIAL_COMBOBOX_STYLE, font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["input"]))

    def _build_ui(self) -> None:
        has_brand_header = bool(self.brand_name or self.subtitle)
        toolbar_row = 1 if has_brand_header else 0
        capture_row = toolbar_row + 1
        content_row = capture_row + 1
        bottom_row = content_row + 1

        self.root.columnconfigure(0, weight=48, minsize=540)
        self.root.columnconfigure(1, weight=52, minsize=570)
        self.root.rowconfigure(content_row, weight=1)

        if has_brand_header:
            self._build_brand_header(row=0)

        toolbar = ttk.Frame(self.root, padding=(6, 4))
        toolbar.grid(row=toolbar_row, column=0, columnspan=2, sticky="ew")
        ttk.Label(toolbar, text="串口", style=SERIAL_CONTROL_LABEL_STYLE).grid(row=0, column=0, padx=(0, 2))
        self.port_combo = ttk.Combobox(toolbar, textvariable=self.port_var, width=18, state="readonly", style=SERIAL_COMBOBOX_STYLE)
        self.port_combo.grid(row=0, column=1, padx=2)
        self.refresh_button = ttk.Button(toolbar, text="刷新串口", command=self.refresh_ports, style=SERIAL_SECONDARY_BUTTON_STYLE)
        self.refresh_button.grid(row=0, column=2, padx=2)
        self.connect_button = ttk.Button(toolbar, text="连接", command=self.connect, style=SERIAL_PRIMARY_BUTTON_STYLE)
        self.connect_button.grid(row=0, column=3, padx=2)
        ttk.Button(toolbar, text="断开", command=self.disconnect, style=SERIAL_DANGER_BUTTON_STYLE).grid(row=0, column=4, padx=2)
        ttk.Button(toolbar, text="空载校准", command=self.calibrate_empty, style=SERIAL_SECONDARY_BUTTON_STYLE).grid(row=0, column=5, padx=2)
        ttk.Button(toolbar, text="重置识别状态", command=self.reset_recognition_state, style=SERIAL_SECONDARY_BUTTON_STYLE).grid(row=0, column=6, padx=2)
        ttk.Button(toolbar, text="查看模型详情", command=self.show_model_details, style=SERIAL_SECONDARY_BUTTON_STYLE).grid(row=0, column=7, padx=2)
        ttk.Label(toolbar, text="方向", style=SERIAL_CONTROL_LABEL_STYLE).grid(row=0, column=8, padx=(8, 2))
        self.orientation_combo = ttk.Combobox(toolbar, textvariable=self.orientation_var, values=list(ORIENTATION_MODES), width=12, state="readonly", style=SERIAL_COMBOBOX_STYLE)
        self.orientation_combo.grid(row=0, column=9, padx=2)
        self.orientation_combo.bind("<<ComboboxSelected>>", self._on_orientation_changed)

        self._build_capture_panel(row=capture_row)

        left = ttk.Frame(self.root, padding=(8, 6, 4, 6))
        left.grid(row=content_row, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="前", anchor="center", style=SERIAL_DIRECTION_LABEL_STYLE).grid(row=0, column=0, sticky="ew", pady=(0, 2))
        canvas_bg = str(self.root.cget("bg"))
        self.canvas = tk.Canvas(left, width=480, height=480, bg=canvas_bg, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.canvas.bind("<Configure>", lambda _event: self._draw_heatmap(self.current_frame))
        ttk.Label(left, text="后", anchor="center", style=SERIAL_DIRECTION_LABEL_STYLE).grid(row=2, column=0, sticky="ew", pady=(2, 0))

        stats = ttk.Frame(left)
        stats.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(stats, text="总压力", style=SERIAL_STATS_LABEL_STYLE).grid(row=0, column=0, padx=8)
        ttk.Label(stats, textvariable=self.total_var, style=SERIAL_STATS_LABEL_STYLE).grid(row=0, column=1, padx=8)
        ttk.Label(stats, text="最大压力", style=SERIAL_STATS_LABEL_STYLE).grid(row=0, column=2, padx=8)
        ttk.Label(stats, textvariable=self.max_var, style=SERIAL_STATS_LABEL_STYLE).grid(row=0, column=3, padx=8)
        ttk.Label(stats, text="活跃点", style=SERIAL_STATS_LABEL_STYLE).grid(row=0, column=4, padx=8)
        ttk.Label(stats, textvariable=self.active_var, style=SERIAL_STATS_LABEL_STYLE).grid(row=0, column=5, padx=8)
        ttk.Label(stats, text="低压力 —— 高压力", style=SERIAL_HELP_LABEL_STYLE).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(3, 0))
        ttk.Label(stats, textvariable=self.max_adc_var, style=SERIAL_HELP_LABEL_STYLE).grid(row=1, column=2, columnspan=2, sticky="w", padx=8, pady=(3, 0))
        ttk.Label(stats, text="压力值为ADC响应强度", style=SERIAL_HELP_LABEL_STYLE).grid(row=1, column=4, columnspan=2, sticky="w", padx=8, pady=(3, 0))

        right = ttk.Frame(self.root, padding=(6, 6, 10, 6))
        right.grid(row=content_row, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self._build_status_panel(right)

        bottom = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        bottom.grid(row=bottom_row, column=0, columnspan=2, sticky="ew")
        ttk.Label(bottom, textvariable=self.summary_var, style=SERIAL_FOOTER_LABEL_STYLE).grid(row=0, column=0, sticky="w")

    def _build_brand_header(self, row: int) -> None:
        header = ttk.Frame(self.root, padding=(12, 6, 12, 2))
        header.grid(row=row, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)
        if self.brand_name:
            ttk.Label(header, text=self.brand_name, anchor="center", font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["brand_title"], "bold")).grid(row=0, column=0, sticky="ew")
        if self.subtitle:
            subtitle_text = serial_subtitle_text(self.model_version, self.subtitle)
            ttk.Label(header, text=subtitle_text, anchor="center", font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["brand_subtitle"])).grid(row=1, column=0, sticky="ew", pady=(1, 0))

    def _build_capture_panel(self, row: int) -> None:
        capture = ttk.LabelFrame(self.root, text="数据采集", padding=(6, 4), style=SERIAL_SECTION_FRAME_STYLE)
        capture.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 3))
        capture.columnconfigure(0, weight=1)

        action_row = ttk.Frame(capture)
        action_row.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        action_row.columnconfigure(1, weight=1)
        action_row.columnconfigure(8, weight=1)
        ttk.Label(action_row, text="采集标签", style=SERIAL_CONTROL_LABEL_STYLE).grid(row=0, column=0, padx=(0, 3), pady=1)
        self.capture_label_entry = ttk.Entry(action_row, textvariable=self.capture_label_var, width=14, style=SERIAL_ENTRY_STYLE)
        self.capture_label_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=1)
        ttk.Label(action_row, text="次数", style=SERIAL_CONTROL_LABEL_STYLE).grid(row=0, column=2, padx=(0, 3), pady=1)
        self.capture_trial_entry = ttk.Entry(action_row, textvariable=self.capture_trial_var, width=5, style=SERIAL_ENTRY_STYLE)
        self.capture_trial_entry.grid(row=0, column=3, padx=(0, 6), pady=1)
        self.capture_dir_button = ttk.Button(action_row, text="选择保存目录", command=self.choose_capture_directory, style=SERIAL_SECONDARY_BUTTON_STYLE)
        self.capture_dir_button.grid(row=0, column=4, padx=2, pady=1)
        self.capture_start_button = ttk.Button(action_row, text="开始采集", command=self.start_capture, style=SERIAL_PRIMARY_BUTTON_STYLE)
        self.capture_start_button.grid(row=0, column=5, padx=2, pady=1)
        self.capture_start_button.configure(state="disabled")
        self.capture_stop_button = ttk.Button(action_row, text="停止采集", command=self.stop_capture, style=SERIAL_DANGER_BUTTON_STYLE)
        self.capture_stop_button.grid(row=0, column=6, padx=(2, 8), pady=1)
        self._capture_field(action_row, "状态", self.capture_status_var, 0, 7, width=14, value_sticky="ew")
        self._capture_field(action_row, "时长", self.capture_duration_var, 0, 9, width=8)

        stats_row = ttk.Frame(capture)
        stats_row.grid(row=1, column=0, sticky="ew", pady=(0, 2))
        stats_fields = (
            ("有效帧", self.capture_frames_var, 7),
            ("识别结果", self.capture_predictions_var, 7),
            ("原始字节", self.capture_raw_bytes_var, 10),
            ("写入队列", self.capture_queue_var, 6),
            ("丢弃事件", self.capture_dropped_var, 6),
        )
        for index, (label, variable, width) in enumerate(stats_fields):
            stats_row.columnconfigure(index, weight=1, uniform="capture_stats")
            item = ttk.Frame(stats_row)
            item.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0))
            self._capture_field(item, label, variable, 0, 0, width=width)

        path_row = ttk.Frame(capture)
        path_row.grid(row=2, column=0, sticky="ew")
        path_row.columnconfigure(1, weight=3)
        path_row.columnconfigure(3, weight=2)
        self._capture_field(path_row, "保存目录", self.capture_dir_var, 0, 0, width=28, value_sticky="ew")
        self._capture_field(path_row, "最近写入错误", self.capture_error_var, 0, 2, width=24, value_sticky="ew")

    def _capture_field(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
        *,
        width: int = 14,
        columnspan: int = 1,
        value_sticky: str = "w",
    ) -> None:
        ttk.Label(parent, text=label, style=SERIAL_CONTROL_LABEL_STYLE).grid(row=row, column=column, sticky="w", padx=(0, 3), pady=1)
        ttk.Label(parent, textvariable=variable, width=width, style=SERIAL_CONTROL_LABEL_STYLE).grid(
            row=row,
            column=column + 1,
            columnspan=columnspan,
            sticky=value_sticky,
            padx=(0, 6),
            pady=1,
        )

    def _build_status_panel(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        connection_box = ttk.LabelFrame(parent, text="连接信息", padding=(6, 4), style=SERIAL_SECTION_FRAME_STYLE)
        connection_box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        connection_box.columnconfigure(1, weight=1)
        connection_box.columnconfigure(3, weight=1)
        connection_box.columnconfigure(5, weight=1)
        self._field(connection_box, "连接状态", self.connection_var, 0, column=0, width=8)
        self._field(connection_box, "当前端口", self.current_port_var, 0, column=2, width=8)
        self._field(connection_box, "接收FPS", self.receive_fps_var, 0, column=4, width=8)
        self._field(connection_box, "波特率", self.baud_var, 1, column=0, width=8)
        self._field(connection_box, "接收字节数", self.received_bytes_var, 1, column=2, width=8)
        self._field(connection_box, "有效协议帧", self.valid_frames_var, 1, column=4, width=8)
        self._field(connection_box, "无效协议帧", self.invalid_frames_var, 2, column=0, width=8)
        self._field(connection_box, "丢弃字节数", self.discarded_bytes_var, 2, column=2, width=8)
        self._field(connection_box, "帧队列", self.frame_queue_var, 2, column=4, width=8, style=SERIAL_WEAK_INFO_LABEL_STYLE)
        self._field(connection_box, "丢弃帧", self.dropped_queue_var, 3, column=0, width=8, style=SERIAL_WEAK_INFO_LABEL_STYLE)
        self._field(connection_box, "串口错误", self.serial_error_var, 3, column=2, width=16, value_columnspan=3, style=SERIAL_WEAK_INFO_LABEL_STYLE)

        result_box = ttk.LabelFrame(parent, text="识别信息", padding=(6, 4), style=SERIAL_SECTION_FRAME_STYLE)
        result_box.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        result_box.columnconfigure(1, weight=1)
        result_box.columnconfigure(2, minsize=12)
        result_box.columnconfigure(4, weight=1)
        result_box.rowconfigure(11, weight=1)
        result_box.rowconfigure(13, weight=1)
        result_box.bind("<Configure>", self._update_recognition_wraps)

        ttk.Label(result_box, text="当前状态", style=SERIAL_SUMMARY_TITLE_STYLE).grid(row=0, column=0, sticky="w")
        ttk.Label(result_box, textvariable=self.summary_status_var, style=SERIAL_SUMMARY_META_STYLE).grid(row=0, column=1, sticky="ew", padx=(6, 10))
        ttk.Label(result_box, text="当前姿势", style=SERIAL_SUMMARY_TITLE_STYLE).grid(row=0, column=3, sticky="w")
        ttk.Label(result_box, textvariable=self.summary_posture_var, style=SERIAL_SUMMARY_POSTURE_STYLE).grid(
            row=0,
            column=4,
            sticky="ew",
            padx=(6, 0),
            pady=(0, 0),
        )
        ttk.Label(result_box, text="置信度", style=SERIAL_SUMMARY_TITLE_STYLE).grid(row=1, column=0, sticky="w", pady=(0, 4))
        ttk.Label(result_box, textvariable=self.summary_confidence_var, style=SERIAL_SUMMARY_META_STYLE).grid(row=1, column=1, sticky="ew", padx=(6, 10), pady=(0, 4))
        ttk.Label(result_box, text="边界状态", style=SERIAL_SUMMARY_TITLE_STYLE).grid(row=1, column=3, sticky="w", pady=(0, 4))
        ttk.Label(result_box, textvariable=self.summary_boundary_var, style=SERIAL_SUMMARY_META_STYLE).grid(row=1, column=4, sticky="ew", padx=(6, 0), pady=(0, 4))

        detail_start = 2
        self._field(result_box, "当前系统状态", self.state_var, detail_start + 0, width=14)
        self._field(result_box, "占用状态", self.occupancy_var, detail_start + 1, width=14)
        self._field(result_box, "姿势", self.posture_var, detail_start + 2, width=14)
        self._field(result_box, "原始候选", self.raw_var, detail_start + 3, width=14, wraplength=210)
        self._field(result_box, "置信度", self.confidence_var, detail_start + 4, width=14)
        self._field(result_box, "第二候选", self.second_var, detail_start + 5, width=14)
        self._field(result_box, "置信差值", self.margin_var, detail_start + 6, width=14)
        self._field(result_box, "识别错误", self.recognition_error_var, detail_start + 7, width=14, wraplength=210)
        self._field(result_box, "当前帧序号", self.frame_index_var, detail_start + 0, column=3, width=14)
        self._field(result_box, "连接运行时间", self.uptime_var, detail_start + 1, column=3, width=14)
        self._field(result_box, "最近推理耗时", self.inference_var, detail_start + 2, column=3, width=14)
        self._field(result_box, "平均推理耗时", self.average_inference_var, detail_start + 3, column=3, width=14)
        self._long_text_field(result_box, "边界原因", self.boundary_reason_var, detail_start + 9)
        self._long_text_field(result_box, "原型诊断", self.prototype_var, detail_start + 11)

    def _field(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        *,
        column: int = 0,
        width: int = 16,
        value_columnspan: int = 1,
        style: str = SERIAL_INFO_LABEL_STYLE,
        wraplength: int | None = None,
    ) -> None:
        ttk.Label(parent, text=label, style=style).grid(row=row, column=column, sticky="w", pady=0)
        value_kwargs: dict[str, object] = {"textvariable": variable, "width": width, "style": SERIAL_INFO_VALUE_STYLE}
        if wraplength is not None:
            value_kwargs["wraplength"] = wraplength
        ttk.Label(parent, **value_kwargs).grid(
            row=row,
            column=column + 1,
            columnspan=value_columnspan,
            sticky="ew",
            padx=(6, 10),
            pady=0,
        )

    def _long_text_field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label, style=SERIAL_INFO_LABEL_STYLE).grid(
            row=row,
            column=0,
            columnspan=5,
            sticky="ew",
            pady=(6, 1),
        )
        value = ttk.Label(
            parent,
            textvariable=variable,
            style=SERIAL_INFO_VALUE_STYLE,
            justify="left",
            anchor="nw",
            wraplength=520,
        )
        value.grid(row=row + 1, column=0, columnspan=5, sticky="nsew", pady=(0, 2))
        self._wrap_labels.append(value)

    def _update_recognition_wraps(self, event: tk.Event | None = None) -> None:
        width = max(int(getattr(event, "width", 0) or 0), 0)
        if width <= 1:
            return
        wraplength = max(width - 28, 320)
        for label in self._wrap_labels:
            label.configure(wraplength=wraplength)

    def refresh_ports(self) -> None:
        try:
            ports = list_serial_ports()
        except Exception as exc:
            self.summary_var.set(f"扫描串口失败：{exc}")
            self.port_combo.configure(values=[])
            return
        devices = [str(getattr(port, "device", port)) for port in ports]
        self.port_combo.configure(values=devices)
        if not devices:
            self.summary_var.set("未找到可用的USB串口设备。")
            self.port_var.set("")
            return
        if self.port_var.get() not in devices:
            self.port_var.set(_recommended_port(devices) or devices[0])
        self.summary_var.set(f"已扫描到 {len(devices)} 个串口。")

    def connect(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("未选择串口", "未选择串口。")
            return
        if self.connecting:
            messagebox.showinfo("正在连接", "正在连接串口，请稍候。")
            return
        if self.reader is not None and self.worker is not None:
            messagebox.showinfo("已连接", "当前已经连接。")
            return
        self.connecting = True
        self.connection_var.set("连接中")
        self.summary_var.set("正在加载模型并打开串口...")
        threading.Thread(target=self._connect_background, args=(port,), name="SerialGuiConnect", daemon=True).start()

    def _connect_background(self, port: str) -> None:
        reader: SerialFrameReader | None = None
        worker: RecognitionWorker | None = None
        try:
            recognizer = self.recognizer or load_runtime_recognizer(model_version=self.model_version)
            reset = getattr(recognizer, "reset", None)
            if callable(reset):
                reset()
            result_queue: Queue[SerialRecognitionResult] = Queue(maxsize=5)
            reader = SerialFrameReader(port=port, baudrate=DEFAULT_SERIAL_BAUDRATE, timeout=0.1, queue_size=5)
            connection_start_time = time.monotonic()
            reader.start()
            worker = RecognitionWorker(
                frame_source=reader,
                recognizer=recognizer,
                result_queue=result_queue,
                orientation_mode=self.orientation_state.get,
                connection_start_time=connection_start_time,
                poll_timeout=0.02,
            )
            worker.start()
        except CsvGuiError as exc:
            if reader is not None:
                reader.stop()
            self.root.after(0, lambda exc=exc: self._finish_connect_error("模型加载失败", str(exc)))
            return
        except Exception as exc:
            if worker is not None:
                worker.stop()
            if reader is not None:
                reader.stop()
            self.root.after(0, lambda exc=exc: self._finish_connect_error("串口连接失败", _friendly_serial_error(exc)))
            return
        self.root.after(
            0,
            lambda: self._finish_connect_success(
                port=port,
                recognizer=recognizer,
                reader=reader,
                worker=worker,
                result_queue=result_queue,
                connection_start_time=connection_start_time,
            ),
        )

    def _finish_connect_success(
        self,
        *,
        port: str,
        recognizer: object,
        reader: SerialFrameReader,
        worker: RecognitionWorker,
        result_queue: Queue[SerialRecognitionResult],
        connection_start_time: float,
    ) -> None:
        self.connecting = False
        if self.reader is not None or self.worker is not None:
            worker.stop()
            reader.stop()
            return
        self.recognizer = recognizer
        self.reader = reader
        self.worker = worker
        self.result_queue = result_queue
        self.connection_start_time = connection_start_time
        self._reported_reader_error = None
        self._reported_worker_error = None
        self.connection_var.set("已连接")
        self.current_port_var.set(port)
        self.summary_var.set("已连接，等待压力帧。")
        self._set_capture_inputs_enabled(True)
        self._schedule_poll()

    def _finish_connect_error(self, title: str, message: str) -> None:
        self.connecting = False
        self.connection_var.set("未连接")
        self.summary_var.set(message)
        self._set_capture_inputs_enabled(True)
        messagebox.showerror(title, message)

    def _on_orientation_changed(self, _event: tk.Event | None = None) -> None:
        self.orientation_state.set(self.orientation_var.get())

    def choose_capture_directory(self) -> None:
        if self._is_recording():
            messagebox.showinfo("正在采集", "采集期间不能修改保存目录。")
            return
        selected = filedialog.askdirectory(title="选择串口采集保存目录")
        if not selected:
            return
        self.capture_output_root = Path(selected)
        self.capture_dir_var.set(str(self.capture_output_root))

    def start_capture(self) -> None:
        if self._is_recording():
            messagebox.showinfo("正在采集", "当前已有采集任务正在运行。")
            return
        if self.reader is None or self.worker is None:
            messagebox.showwarning("未连接", "请先连接串口。")
            return
        if self.reader.valid_frames <= 0:
            messagebox.showwarning("尚未收到有效压力帧", "尚未收到有效压力帧。")
            return
        if self.capture_output_root is None:
            messagebox.showwarning("未选择保存目录", "请先选择保存目录。")
            return
        label = self.capture_label_var.get().strip()
        if not label:
            messagebox.showwarning("标签为空", "采集标签不能为空。")
            return
        try:
            trial = int(self.capture_trial_var.get().strip())
            if trial <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("次数不合法", "采集次数必须是正整数。")
            return

        recorder = SerialDataRecorder()
        try:
            capture_dir = recorder.start(
                output_root=self.capture_output_root,
                label=label,
                trial=trial,
                serial_port=self.current_port_var.get(),
                baudrate=DEFAULT_SERIAL_BAUDRATE,
                orientation=self.orientation_var.get(),
                model_version=self.model_version,
                serial_reader_stats_start=self.reader.stats(),
            )
        except Exception as exc:
            self.recorder = recorder
            self.capture_status_var.set("保存失败")
            self.capture_error_var.set(str(exc))
            messagebox.showerror("保存失败", f"采集保存失败：{exc}")
            return

        try:
            self.reader.begin_recording_boundary(
                raw_chunk_listener=recorder.record_raw_chunk,
                parsed_frame_listener=recorder.record_parsed_frame,
            )
        except Exception as exc:
            recorder.stop(serial_reader_stats_end=self.reader.stats())
            self.recorder = None
            self.capture_status_var.set("保存失败")
            self.capture_error_var.set(str(exc))
            messagebox.showerror("保存失败", f"采集边界同步失败：{exc}")
            return

        self.recorder = recorder
        self.worker.prediction_listener = recorder.record_prediction
        self.capture_status_var.set(f"正在采集：{label} · 第{trial}次")
        self.capture_dir_var.set(str(capture_dir))
        self.capture_error_var.set("—")
        self.summary_var.set(f"正在采集：{label} · 第{trial}次")
        self._set_capture_inputs_enabled(False)

    def stop_capture(self) -> None:
        self._stop_capture(show_message=True)

    def disconnect(self) -> None:
        self._stop_capture(show_message=False)
        self.connecting = False
        if self.poll_after_id:
            self.root.after_cancel(self.poll_after_id)
            self.poll_after_id = None
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.result_queue is not None:
            _drain_queue(self.result_queue)
            self.result_queue = None
        self.connection_var.set("未连接")
        self.current_port_var.set("—")
        self.frame_queue_var.set("0")
        self._set_capture_inputs_enabled(True)
        self.summary_var.set("已断开。")

    def _stop_capture(self, *, show_message: bool) -> None:
        recorder = self.recorder
        if recorder is None or not recorder.is_recording:
            if show_message:
                messagebox.showinfo("未采集", "当前没有正在进行的采集。")
            return
        self._detach_recorder_hooks()
        stats_end = self.reader.stats() if self.reader is not None else {}
        recorder.stop(serial_reader_stats_end=stats_end)
        stats = recorder.stats()
        if stats["last_error"] is not None or not stats["data_complete"]:
            self.capture_status_var.set("保存失败")
            self.capture_error_var.set("采集数据不完整。" if stats["last_error"] is None else str(stats["last_error"]))
        else:
            self.capture_status_var.set("已完成")
            self.capture_error_var.set("—")
        self._refresh_capture_stats()
        self._set_capture_inputs_enabled(True)
        message = (
            f"采集完成。保存路径：{recorder.capture_dir}；"
            f"有效帧：{stats['valid_frames_saved']}；持续时间：{stats['duration_s']:.2f}s；"
            f"可直接查看的串口数据：{SERIAL_TEXT_FILENAME}。"
        )
        self.summary_var.set(message)
        if show_message:
            messagebox.showinfo("采集完成", message)

    def _detach_recorder_hooks(self) -> None:
        if self.reader is not None:
            self.reader.raw_chunk_listener = None
            self.reader.parsed_frame_listener = None
        if self.worker is not None:
            self.worker.prediction_listener = None

    def _is_recording(self) -> bool:
        return bool(self.recorder is not None and self.recorder.is_recording)

    def _set_capture_inputs_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        readonly_state = "readonly" if enabled else "disabled"
        for widget in (
            self.capture_label_entry,
            self.capture_trial_entry,
            self.capture_dir_button,
            self.refresh_button,
            self.connect_button,
        ):
            widget.configure(state=state)
        start_state = "normal" if enabled and self.reader is not None and self.worker is not None else "disabled"
        self.capture_start_button.configure(state=start_state)
        self.port_combo.configure(state=readonly_state)
        self.orientation_combo.configure(state=readonly_state)

    def calibrate_empty(self) -> None:
        if self.worker is None or self.reader is None:
            messagebox.showwarning("未连接", "必须先连接串口。")
            return
        if self.current_frame is None:
            messagebox.showwarning("无法校准", "尚未收到有效压力帧，无法校准。")
            return
        ok = messagebox.askokcancel("空载校准", "请确认坐垫上没有人、物品或其他负载，再执行空载校准。")
        if not ok:
            return
        try:
            self.worker.calibrate(frame=self.current_frame, wait=True, timeout=2.0)
            self.worker.reset_recognizer(wait=True, timeout=2.0)
            if self.result_queue is not None:
                _drain_queue(self.result_queue)
            self.summary_var.set("空载校准完成，请坐下开始识别。")
        except Exception as exc:
            messagebox.showerror("校准失败", f"空载校准失败：{exc}")

    def reset_recognition_state(self) -> None:
        if self.worker is None:
            messagebox.showwarning("未连接", "必须先连接串口。")
            return
        try:
            self.worker.reset_recognizer(wait=True, timeout=2.0)
            if self.result_queue is not None:
                _drain_queue(self.result_queue)
            self.summary_var.set("识别状态已重置。")
        except Exception as exc:
            messagebox.showerror("重置失败", f"识别状态重置失败：{exc}")

    def show_model_details(self) -> None:
        if self.model_details_window is not None and self.model_details_window.winfo_exists():
            self.model_details_window.lift()
            self.model_details_window.focus_force()
            return
        try:
            if self.recognizer is None:
                self.recognizer = load_runtime_recognizer(model_version=self.model_version)
        except Exception as exc:
            messagebox.showerror("模型加载失败", f"模型加载失败：{exc}")
            return
        details = {
            "display_name": model_version_display_name(self.model_version),
            "loaded_model_version": self.model_version,
            **model_runtime_versions(self.recognizer),
            **model_export_info(self.recognizer),
        }
        details_text = json.dumps(details, ensure_ascii=False, indent=2)
        window = tk.Toplevel(self.root)
        self.model_details_window = window
        window.title("模型详情")
        window.geometry("820x560")
        window.minsize(640, 420)

        def close_window(_event: tk.Event | None = None) -> None:
            if self.model_details_window is window:
                self.model_details_window = None
            if window.winfo_exists():
                window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)
        window.bind("<Escape>", close_window)
        container = ttk.Frame(window, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)
        summary = (
            f"{details['display_name']}\n"
            f"当前加载版本：{details['loaded_model_version']}\n"
            f"父模型版本：{details.get('parent_model_version') or '—'}\n"
            f"后靠子模型版本：{details.get('submodel_version') or '—'}\n"
            f"侧向子模型版本：{details.get('lateral_submodel_version') or '—'}"
        )
        ttk.Label(container, text=summary, justify="left").grid(row=0, column=0, sticky="ew", pady=(0, 8))
        text = scrolledtext.ScrolledText(container, wrap="word", width=96, height=24)
        text.grid(row=1, column=0, sticky="nsew")
        text.insert("1.0", details_text)
        text.configure(state="disabled")
        ttk.Button(container, text="关闭", command=close_window).grid(row=2, column=0, sticky="e", pady=(10, 0))

    def _schedule_poll(self) -> None:
        if self.poll_after_id:
            self.root.after_cancel(self.poll_after_id)
        self.poll_after_id = self.root.after(50, self._poll)

    def _poll(self) -> None:
        self.poll_after_id = None
        self._poll_results()
        self._refresh_runtime_stats()
        self._report_background_errors()
        if self.reader is not None or self.worker is not None:
            self._schedule_poll()

    def _poll_results(self) -> None:
        if self.result_queue is None:
            return
        latest: SerialRecognitionResult | None = None
        while True:
            try:
                latest = self.result_queue.get_nowait()
            except Empty:
                break
        if latest is not None:
            self._render_result(latest)

    def _render_result(self, result: SerialRecognitionResult) -> None:
        record = result.prediction
        self.current_frame = result.frame
        self.current_record = record
        self._draw_heatmap(result.frame)
        self.total_var.set(f"{record.total_pressure:.1f}")
        self.max_var.set(f"{record.max_pressure:.1f}")
        self.max_adc_var.set(f"当前最大ADC：{int(round(record.max_pressure))}")
        self.active_var.set(str(record.active_points))
        self.state_var.set(STATE_LABELS.get(record.display_status, record.display_status))
        self.occupancy_var.set(_display_occupancy(record.occupancy_state))
        posture_text = record.posture or ("边界姿势 / 低置信度" if record.display_status == "POSTURE" and record.is_boundary else "—")
        self.posture_var.set(posture_text)
        self.raw_var.set(_format_raw_candidate(record))
        confidence_text = _format_optional(record.posture_confidence)
        self.confidence_var.set(confidence_text)
        self.second_var.set(record.second_label or "—")
        self.margin_var.set(_format_optional(record.margin))
        self.boundary_var.set("是" if record.is_boundary else "否，稳定识别")
        self.boundary_reason_var.set(record.boundary_reason or record.lateral_boundary_reasons or "—")
        self.prototype_var.set(_prototype_label(record))
        self.frame_index_var.set(str(record.frame_index))
        self.uptime_var.set(f"{record.timestamp:.2f}s")
        self.inference_var.set(f"{result.inference_ms:.1f} ms")
        if self.worker and self.worker.average_inference_ms is not None:
            self.average_inference_var.set(f"{self.worker.average_inference_ms:.1f} ms")
        self.summary_status_var.set(_summary_state_label(record))
        self.summary_posture_var.set(posture_text)
        self.summary_confidence_var.set(confidence_text)
        self.summary_boundary_var.set("是" if record.is_boundary else "否")

    def _refresh_runtime_stats(self) -> None:
        if self.reader is not None:
            stats = self.reader.stats()
            self.received_bytes_var.set(str(stats["received_bytes"]))
            self.valid_frames_var.set(str(stats["valid_frames"]))
            self.invalid_frames_var.set(str(stats["invalid_frames"]))
            self.discarded_bytes_var.set(str(stats["discarded_bytes"]))
            self.receive_fps_var.set(f"{float(stats['current_fps']):.1f}")
            self.dropped_queue_var.set(str(stats["dropped_queue_frames"]))
            self.serial_error_var.set("—" if stats["last_error"] is None else str(stats["last_error"]))
            queue_obj = getattr(self.reader, "_queue", None)
            self.frame_queue_var.set("—" if queue_obj is None else str(queue_obj.qsize()))
        if self.worker is not None and self.worker.last_error is not None:
            self.recognition_error_var.set(str(self.worker.last_error))
        elif self.worker is not None:
            self.recognition_error_var.set("—")
        self._refresh_capture_stats()

    def _refresh_capture_stats(self) -> None:
        if self.recorder is None:
            return
        stats = self.recorder.stats()
        self.capture_duration_var.set(f"{float(stats['duration_s']):.2f}s")
        self.capture_raw_bytes_var.set(str(stats["raw_bytes_saved"]))
        self.capture_frames_var.set(str(stats["valid_frames_saved"]))
        self.capture_predictions_var.set(str(stats["predictions_saved"]))
        self.capture_queue_var.set(str(stats["recorder_queue_size"]))
        self.capture_dropped_var.set(str(stats["recorder_dropped_events"]))
        if stats["last_error"] is not None:
            self.capture_status_var.set("保存失败")
            self.capture_error_var.set(str(stats["last_error"]))
        elif int(stats["recorder_dropped_events"]) > 0:
            self.capture_error_var.set("采集队列溢出，数据可能不完整。")
        elif self._is_recording():
            self.capture_error_var.set("—")

    def _report_background_errors(self) -> None:
        if self.reader is not None and self.reader.last_error is not None:
            message = str(self.reader.last_error)
            if message != self._reported_reader_error:
                self._reported_reader_error = message
                self.summary_var.set("设备连接已断开或串口读取发生错误。")
                self.summary_status_var.set("错误")
                messagebox.showerror("串口错误", "设备连接已断开或串口读取发生错误。")
        if self.worker is not None and self.worker.last_error is not None:
            message = str(self.worker.last_error)
            if message != self._reported_worker_error:
                self._reported_worker_error = message
                self.summary_var.set(f"识别线程发生错误：{message}")
                self.summary_status_var.set("错误")
                messagebox.showerror("识别错误", f"识别线程发生错误：{message}")

    def _draw_heatmap(self, frame: np.ndarray | None) -> None:
        self.canvas.delete("all")
        if frame is None:
            frame = np.zeros((16, 16), dtype=np.float32)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = int(float(self.canvas.cget("width")))
        if canvas_height <= 1:
            canvas_height = int(float(self.canvas.cget("height")))
        geometry = heatmap_grid_geometry(canvas_width, canvas_height)
        max_value = float(np.max(frame)) if frame.size else 0.0
        for row in range(16):
            for col in range(16):
                x0, y0, x1, y1 = geometry.bounds_for(row, col)
                value = float(frame[row, col])
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=pressure_to_color(value, max_value),
                    outline="#172033",
                )
        left_label, right_label = _heatmap_side_label_positions(geometry)
        self.canvas.create_text(*left_label, text="左", anchor="e", font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["direction"], "bold"))
        self.canvas.create_text(*right_label, text="右", anchor="w", font=(SERIAL_FONT_FAMILY, SERIAL_FONT_SIZES["direction"], "bold"))

    def _call_recognizer_reset(self) -> None:
        reset = getattr(self.recognizer, "reset", None)
        if callable(reset):
            reset()

    def _on_close(self) -> None:
        self.disconnect()
        self.root.destroy()


def _recommended_port(devices: list[str]) -> str | None:
    for device in devices:
        if _is_windows_com_port(device):
            return device
    preferred = ("usbserial", "wchusbserial", "SLAB_USBtoUART", "usbmodem")
    rejected = ("Bluetooth-Incoming-Port", "debug-console")
    for needle in preferred:
        for device in devices:
            if needle in device and not any(item in device for item in rejected):
                return device
    for device in devices:
        if not any(item in device for item in rejected):
            return device
    return devices[0] if devices else None


def _is_windows_com_port(device: str) -> bool:
    candidate = str(device).upper()
    return candidate.startswith("COM") and candidate[3:].isdigit()


class _ThreadSafeValue:
    def __init__(self, value: str) -> None:
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            return self._value

    def set(self, value: str) -> None:
        with self._lock:
            self._value = value


def _friendly_serial_error(exc: BaseException) -> str:
    message = str(exc)
    lowered = message.lower()
    if "busy" in lowered or "resource busy" in lowered or "permission" in lowered:
        return "串口被占用，请先关闭CoolTerm。"
    if "no such file" in lowered or "not found" in lowered or "could not open port" in lowered:
        return "无法打开所选串口。"
    return f"无法打开所选串口：{message}"


def _drain_queue(queue_obj: Queue[object]) -> None:
    while True:
        try:
            queue_obj.get_nowait()
        except Empty:
            return


def _format_optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _display_occupancy(value: str | None) -> str:
    mapping = {
        "HUMAN": "有人",
        "EMPTY": "空载",
        "OBJECT": "物品",
        "UNKNOWN": "未知",
    }
    if not value:
        return "—"
    return mapping.get(str(value), str(value))


def _heatmap_side_label_positions(geometry: object, gap: float = 10.0) -> tuple[tuple[float, float], tuple[float, float]]:
    offset_x = float(getattr(geometry, "offset_x"))
    offset_y = float(getattr(geometry, "offset_y"))
    square_size = float(getattr(geometry, "square_size"))
    canvas_width = float(getattr(geometry, "canvas_width"))
    center_y = offset_y + square_size / 2.0
    left_x = max(2.0, offset_x - gap)
    right_x = min(canvas_width - 2.0, offset_x + square_size + gap)
    return (left_x, center_y), (right_x, center_y)


def _summary_state_label(record: FramePrediction) -> str:
    if record.is_boundary:
        return "边界姿势"
    if record.display_status == "EMPTY" or record.occupancy_state == "EMPTY":
        return "空载"
    if record.display_status == "ERROR":
        return "错误"
    if record.display_status == "POSTURE" and record.posture:
        return "稳定识别"
    return "正在稳定"


def _format_raw_candidate(record: FramePrediction) -> str:
    if record.raw_label:
        return f"{record.raw_label} {_format_optional(record.raw_confidence)}"
    return "—"


def _prototype_label(record: FramePrediction) -> str:
    if record.lateral_prototype_label:
        return f"lateral={record.lateral_prototype_label}; d={_format_optional(record.lateral_prototype_distance)}"
    if record.prototype_diagnosis:
        return record.prototype_diagnosis
    if record.raw_label or record.second_label:
        return f"second={record.second_label or '—'}"
    return "—"


def main(
    argv: Sequence[str] | None = None,
    *,
    app_title: str | None = None,
    brand_name: str | None = None,
    subtitle: str | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="macOS realtime serial posture recognition app.")
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        choices=[
            "v1",
            "v2_candidate",
            "v2_1_candidate",
            "v2_2_candidate",
            "v2_3_candidate",
            "v2_3_1_candidate",
            "v2_4_candidate",
            "v2_4_1_candidate",
            "v2_4_2_candidate",
            "v2_4_3_candidate",
        ],
        help="Recognizer model version to load. The macOS realtime app defaults to v2_4_3_candidate.",
    )
    args = parser.parse_args(argv)
    root = tk.Tk()
    PostureSerialApp(root, model_version=args.model_version, app_title=app_title, brand_name=brand_name, subtitle=subtitle)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
