from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Sequence

import numpy as np

from .csv_gui import HeatmapGridGeometry, heatmap_grid_geometry, model_version_display_name
from .csv_gui_core import load_runtime_recognizer, model_export_info
from .gui import pressure_to_color
from .offline_posture_analyzer import (
    DEFAULT_MODEL_VERSION,
    OfflineAnalysisResult,
    OfflinePostureAnalyzer,
    export_offline_analysis,
    posture_statistics,
    resolve_fps,
    resolve_orientation,
)
from .offline_serial_parser import parse_serial_input, select_serial_input
from .serial_gui_core import ORIENTATION_MODES


APP_NAME = "离线串口坐姿分析软件"
DEFAULT_WINDOW_GEOMETRY = "1240x840"
DEFAULT_WINDOW_SIZE = (1240, 840)
MIN_WINDOW_SIZE = (1100, 700)
SCREEN_HORIZONTAL_SAFE_MARGIN = 80
SCREEN_VERTICAL_SAFE_MARGIN = 100
RIGHT_NOTEBOOK_MIN_HEIGHT = 320
OFFLINE_FONT_FAMILY = "TkDefaultFont"
OFFLINE_FONT_SIZES = {
    "brand_title": 20,
    "brand_subtitle": 13,
    "section_title": 16,
    "field": 13,
    "field_value": 13,
    "button": 12,
    "input": 12,
    "direction": 14,
    "stats": 13,
    "helper": 11,
    "footer": 12,
}


@dataclass(frozen=True)
class WindowGeometry:
    width: int
    height: int
    x: int
    y: int

    def as_tk_geometry(self) -> str:
        return f"{self.width}x{self.height}+{self.x}+{self.y}"


def calculate_initial_window_geometry(screen_width: int, screen_height: int) -> WindowGeometry:
    min_width, min_height = MIN_WINDOW_SIZE
    requested_width, requested_height = DEFAULT_WINDOW_SIZE
    safe_width = max(min_width, int(screen_width) - SCREEN_HORIZONTAL_SAFE_MARGIN)
    safe_height = max(min_height, int(screen_height) - SCREEN_VERTICAL_SAFE_MARGIN)
    width = min(requested_width, safe_width)
    height = min(requested_height, safe_height)
    x = max(0, (int(screen_width) - width) // 2)
    y = max(0, (int(screen_height) - height) // 2)
    return WindowGeometry(width=width, height=height, x=x, y=y)


def _mousewheel_scroll_units(event: object) -> int:
    number = getattr(event, "num", None)
    if number == 4:
        return -1
    if number == 5:
        return 1
    delta = int(getattr(event, "delta", 0) or 0)
    if delta == 0:
        return 0
    magnitude = max(1, abs(delta) // 120)
    return -magnitude if delta > 0 else magnitude


class PostureOfflineSerialApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        model_version: str = DEFAULT_MODEL_VERSION,
        brand_name: str | None = None,
        subtitle: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self.root = root
        self.model_version = model_version
        self.brand_name = brand_name
        self.subtitle = subtitle or APP_NAME
        self.app_title = app_title or APP_NAME
        self.selected_path: Path | None = None
        self.analysis_result: OfflineAnalysisResult | None = None
        self.event_queue: Queue[tuple[str, Any]] = Queue()
        self.cancel_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.after_id: str | None = None
        self.play_after_id: str | None = None
        self.playing = False
        self.current_frame_index = 0
        self._global_mousewheel_bound = False

        self._init_vars()
        self._configure_root()
        self._configure_styles()
        self._build_ui()
        self._bind_global_mousewheel()
        self._set_idle_controls()
        self._draw_empty_heatmap()
        self.after_id = self.root.after(100, self._poll_worker)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_vars(self) -> None:
        self.path_var = tk.StringVar(value="未选择")
        self.input_file_var = tk.StringVar(value="—")
        self.input_directory_var = tk.StringVar(value="—")
        self.input_type_var = tk.StringVar(value="—")
        self.file_size_var = tk.StringVar(value="—")
        self.metadata_path_var = tk.StringVar(value="—")
        self.manual_label_var = tk.StringVar(value="—")
        self.manual_label_trial_var = tk.StringVar(value="—")
        self.trial_var = tk.StringVar(value="—")
        self.capture_time_var = tk.StringVar(value="—")
        self.orientation_var = tk.StringVar(value="原始")
        self.fps_var = tk.StringVar(value="20")
        self.fps_source_var = tk.StringVar(value="—")
        self.total_bytes_var = tk.StringVar(value="—")
        self.valid_packets_var = tk.StringVar(value="—")
        self.invalid_packets_var = tk.StringVar(value="—")
        self.discarded_bytes_var = tk.StringVar(value="—")
        self.invalid_lines_var = tk.StringVar(value="—")
        self.data_complete_var = tk.StringVar(value="—")
        self.calibration_var = tk.StringVar(value="—")
        self.progress_var = tk.StringVar(value="未开始")
        self.model_var = tk.StringVar(value=model_version_display_name(self.model_version))

        self.overall_var = tk.StringVar(value="—")
        self.status_var = tk.StringVar(value="—")
        self.dominant_var = tk.StringVar(value="—")
        self.share_var = tk.StringVar(value="—")
        self.mean_conf_var = tk.StringVar(value="—")
        self.boundary_rate_var = tk.StringVar(value="—")
        self.human_duration_var = tk.StringVar(value="—")
        self.stable_duration_var = tk.StringVar(value="—")
        self.label_match_var = tk.StringVar(value="—")
        self.warnings_var = tk.StringVar(value="—")

        self.frame_info_var = tk.StringVar(value="当前帧：—")
        self.frame_state_var = tk.StringVar(value="状态：—")
        self.frame_posture_var = tk.StringVar(value="姿势：—")
        self.frame_confidence_var = tk.StringVar(value="置信度：—")
        self.frame_boundary_var = tk.StringVar(value="Boundary：—")
        self.total_pressure_var = tk.StringVar(value="总压力：—")
        self.max_pressure_var = tk.StringVar(value="最大压力：—")
        self.active_points_var = tk.StringVar(value="活跃点：—")

    def _configure_root(self) -> None:
        self.root.title(self.app_title)
        geometry = calculate_initial_window_geometry(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        self.requested_geometry = geometry.as_tk_geometry()
        self.root.geometry(self.requested_geometry)
        self.root.minsize(*MIN_WINDOW_SIZE)
        self.root.resizable(True, True)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Offline.TLabelframe.Label", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["section_title"], "bold"))
        style.configure("Offline.TLabel", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["field"]))
        style.configure("OfflineValue.TLabel", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["field_value"]))
        style.configure("Offline.TButton", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["button"]), padding=(8, 4))
        style.configure("Offline.TEntry", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["input"]))
        style.configure("Offline.TCombobox", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["input"]))
        style.configure("Offline.Treeview", font=(OFFLINE_FONT_FAMILY, 11), rowheight=24)
        style.configure("Offline.Treeview.Heading", font=(OFFLINE_FONT_FAMILY, 11, "bold"))

    def _build_ui(self) -> None:
        if self.brand_name:
            header = ttk.Frame(self.root)
            header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 2))
            ttk.Label(header, text=self.brand_name, font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["brand_title"], "bold")).grid(row=0, column=0, sticky="w")
            ttk.Label(
                header,
                text=f"{self.subtitle} · {model_version_display_name(self.model_version)}",
                font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["brand_subtitle"]),
            ).grid(row=1, column=0, sticky="w")
            controls_row = 1
        else:
            controls_row = 0

        self._build_top_controls(row=controls_row)
        main = ttk.Frame(self.root)
        main.grid(row=2, column=0, sticky="nsew", padx=10, pady=(8, 8))
        main.columnconfigure(0, weight=48)
        main.columnconfigure(1, weight=52)
        main.rowconfigure(0, weight=1)
        self._build_left_panel(main)
        self._build_right_panel(main)

        self.footer = ttk.Label(self.root, textvariable=self.progress_var, style="Offline.TLabel", anchor="w")
        self.footer.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))

    def _build_top_controls(self, *, row: int) -> None:
        box = ttk.LabelFrame(self.root, text="离线分析", style="Offline.TLabelframe")
        box.grid(row=row, column=0, sticky="ew", padx=14, pady=(6, 0))
        for col in range(8):
            box.columnconfigure(col, weight=0)
        box.columnconfigure(5, weight=1)

        self.folder_button = ttk.Button(box, text="选择采集文件夹", command=self.choose_capture_folder, style="Offline.TButton")
        self.file_button = ttk.Button(box, text="选择串口数据", command=self.choose_serial_file, style="Offline.TButton")
        self.start_button = ttk.Button(box, text="开始分析", command=self.start_analysis, style="Offline.TButton")
        self.cancel_button = ttk.Button(box, text="取消分析", command=self.cancel_analysis, style="Offline.TButton")
        self.export_button = ttk.Button(box, text="导出报告", command=self.export_report, style="Offline.TButton")
        self.model_button = ttk.Button(box, text="查看模型详情", command=self.show_model_details, style="Offline.TButton")
        self.orientation_combo = ttk.Combobox(box, textvariable=self.orientation_var, values=ORIENTATION_MODES, state="readonly", width=16, style="Offline.TCombobox")
        self.fps_entry = ttk.Entry(box, textvariable=self.fps_var, width=8, style="Offline.TEntry")

        self.folder_button.grid(row=0, column=0, padx=(6, 4), pady=6)
        self.file_button.grid(row=0, column=1, padx=4, pady=6)
        self.start_button.grid(row=0, column=2, padx=4, pady=6)
        self.cancel_button.grid(row=0, column=3, padx=4, pady=6)
        self.export_button.grid(row=0, column=4, padx=4, pady=6)
        self.model_button.grid(row=0, column=5, padx=4, pady=6)
        ttk.Label(box, text="数据方向", style="Offline.TLabel").grid(row=1, column=0, padx=(6, 4), pady=(0, 6), sticky="e")
        self.orientation_combo.grid(row=1, column=1, padx=4, pady=(0, 6), sticky="w")
        ttk.Label(box, text="FPS", style="Offline.TLabel").grid(row=1, column=2, padx=(10, 4), pady=(0, 6), sticky="e")
        self.fps_entry.grid(row=1, column=3, padx=4, pady=(0, 6), sticky="w")
        ttk.Label(box, text="模型", style="Offline.TLabel").grid(row=1, column=4, padx=(10, 4), pady=(0, 6), sticky="e")
        ttk.Label(box, textvariable=self.model_var, style="OfflineValue.TLabel").grid(row=1, column=5, columnspan=3, sticky="w", padx=(0, 6), pady=(0, 6))

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        heat_box = ttk.LabelFrame(left, text="逐帧压力热力图", style="Offline.TLabelframe")
        heat_box.grid(row=0, column=0, sticky="nsew")
        heat_box.rowconfigure(1, weight=1)
        heat_box.columnconfigure(1, weight=1)
        ttk.Label(heat_box, text="前", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["direction"], "bold")).grid(row=0, column=1, pady=(4, 0))
        ttk.Label(heat_box, text="左", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["direction"], "bold")).grid(row=1, column=0, padx=(6, 6))
        ttk.Label(heat_box, text="右", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["direction"], "bold")).grid(row=1, column=2, padx=(6, 6))
        ttk.Label(heat_box, text="后", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["direction"], "bold")).grid(row=2, column=1, pady=(0, 4))
        self.canvas = tk.Canvas(heat_box, bg="#f4f6f8", highlightthickness=0, width=500, height=360)
        self.canvas.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.canvas.bind("<Configure>", lambda _event: self._render_current_frame())

        stats = ttk.Frame(left)
        stats.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for index, variable in enumerate((self.total_pressure_var, self.max_pressure_var, self.active_points_var)):
            ttk.Label(stats, textvariable=variable, style="OfflineValue.TLabel").grid(row=0, column=index, sticky="w", padx=(0, 18))
        ttk.Label(stats, text="压力值为ADC响应强度", font=(OFFLINE_FONT_FAMILY, OFFLINE_FONT_SIZES["helper"])).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))

        controls = ttk.Frame(left)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for col in range(3):
            controls.columnconfigure(col, weight=0)
        controls.columnconfigure(3, weight=1)
        for text, command, row, col in [
            ("开头", self.jump_start, 0, 0),
            ("上一帧", self.previous_frame, 0, 1),
            ("播放", self.play, 0, 2),
            ("暂停", self.pause, 1, 0),
            ("下一帧", self.next_frame, 1, 1),
            ("结尾", self.jump_end, 1, 2),
        ]:
            ttk.Button(controls, text=text, command=command, style="Offline.TButton").grid(row=row, column=col, padx=3, pady=2, sticky="ew")
        self.frame_scale = ttk.Scale(controls, from_=0, to=0, orient="horizontal", command=self._on_frame_scale)
        self.frame_scale.grid(row=0, column=3, sticky="ew", padx=3, pady=2)
        self.speed_var = tk.StringVar(value="1×")
        ttk.Combobox(controls, textvariable=self.speed_var, values=("0.5×", "1×", "2×", "5×", "最大速度"), state="readonly", width=8).grid(row=1, column=3, sticky="w", padx=3, pady=2)

        frame_info = ttk.Frame(left)
        frame_info.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        for col in range(3):
            frame_info.columnconfigure(col, weight=1)
        for row, var in enumerate((self.frame_info_var, self.frame_state_var, self.frame_posture_var, self.frame_confidence_var, self.frame_boundary_var)):
            ttk.Label(frame_info, textvariable=var, style="OfflineValue.TLabel").grid(row=row // 3, column=row % 3, sticky="w", padx=(0, 12), pady=1)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.right_scroll_container = right

        content = self._build_scrollable_right_page(right)
        self._build_file_info(content)
        self._build_summary(content)
        self._build_tables(content)

    def _build_scrollable_right_page(self, parent: ttk.Frame) -> ttk.Frame:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0, bg="#f4f6f8", yscrollincrement=12)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content.columnconfigure(0, weight=1)
        self.right_scroll_canvas = canvas
        self.right_scrollbar = scrollbar
        self.right_scroll_content = content

        def update_scroll_region(_event: object) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_content(event: object) -> None:
            width = getattr(event, "width", canvas.winfo_width())
            canvas.itemconfigure(window_id, width=width)

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_content)
        return content

    def _bind_global_mousewheel(self) -> None:
        if getattr(self, "_global_mousewheel_bound", False):
            return
        self.root.bind_all("<MouseWheel>", lambda event: PostureOfflineSerialApp._route_mousewheel(self, event), add="+")
        self.root.bind_all("<Button-4>", lambda event: PostureOfflineSerialApp._route_mousewheel(self, event), add="+")
        self.root.bind_all("<Button-5>", lambda event: PostureOfflineSerialApp._route_mousewheel(self, event), add="+")
        self._global_mousewheel_bound = True

    def _unbind_global_mousewheel(self) -> None:
        if not getattr(self, "_global_mousewheel_bound", False):
            return
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")
        self._global_mousewheel_bound = False

    def _route_mousewheel(self, event: object) -> str | None:
        container = getattr(self, "right_scroll_container", None)
        canvas = getattr(self, "right_scroll_canvas", None)
        if container is None or canvas is None:
            return None
        try:
            widget = self.root.winfo_containing(getattr(event, "x_root", 0), getattr(event, "y_root", 0))
        except Exception:
            return None
        if widget is None or not _is_descendant(widget, container):
            return None
        tree = PostureOfflineSerialApp._treeview_for_widget(self, widget)
        if tree is not None:
            return self._scroll_treeview_for_mousewheel(tree, event)
        units = _mousewheel_scroll_units(event)
        if units:
            canvas.yview_scroll(units, "units")
        return "break"

    def _treeview_for_widget(self, widget: object) -> ttk.Treeview | None:
        for tree in (getattr(self, "stats_table", None), getattr(self, "segment_table", None)):
            if tree is not None and _is_descendant(widget, tree):
                return tree
        return None

    def _build_file_info(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="文件与协议信息", style="Offline.TLabelframe")
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(0, weight=0, minsize=118)
        box.columnconfigure(1, weight=1)
        box.columnconfigure(2, weight=0, minsize=118)
        box.columnconfigure(3, weight=1)
        self.file_info_wrap_labels: list[ttk.Label] = []
        self.file_info_wrap_labels.append(self._field(box, "输入文件", self.input_file_var, row=0, column=0, columnspan=3, wraplength=520, pady=1))
        self.file_info_wrap_labels.append(self._field(box, "所在目录", self.input_directory_var, row=1, column=0, columnspan=3, wraplength=520, pady=1))
        rows = [
            [("输入类型", self.input_type_var), ("人工标签与采集次数", self.manual_label_trial_var)],
            [("采集时间", self.capture_time_var), ("FPS", self.fps_var)],
            [("数据方向", self.orientation_var), ("有效协议包", self.valid_packets_var)],
            [("无效协议包", self.invalid_packets_var), ("数据完整状态", self.data_complete_var)],
        ]
        for row_offset, row_fields in enumerate(rows, start=2):
            for index, (label, var) in enumerate(row_fields):
                self._field(box, label, var, row=row_offset, column=index * 2, wraplength=190, pady=1)
        self.file_info_wrap_labels.append(self._field(box, "校准状态", self.calibration_var, row=6, column=0, columnspan=3, wraplength=520, pady=(1, 4)))
        box.bind("<Configure>", lambda event: PostureOfflineSerialApp._update_file_info_wraplength(self, event), add="+")

    def _update_file_info_wraplength(self, event: object | None = None) -> None:
        labels = getattr(self, "file_info_wrap_labels", [])
        width = int(getattr(event, "width", 0) or 0)
        if width <= 0 and labels:
            try:
                width = int(labels[0].master.winfo_width())
            except Exception:
                width = 640
        wraplength = max(320, width - 130)
        for label in labels:
            label.configure(wraplength=wraplength)

    def _build_summary(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="综合结果", style="Offline.TLabelframe")
        box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for col in (1, 3, 5):
            box.columnconfigure(col, weight=1)
        rows = [
            [("综合判定", self.overall_var), ("结果状态", self.status_var), ("人工标签", self.manual_label_var)],
            [("主姿势", self.dominant_var), ("主姿势占比", self.share_var), ("是否一致", self.label_match_var)],
            [("平均置信度", self.mean_conf_var), ("边界率", self.boundary_rate_var), ("人体时长", self.human_duration_var)],
            [("稳定识别时长", self.stable_duration_var)],
        ]
        for row, row_fields in enumerate(rows):
            for index, (label, var) in enumerate(row_fields):
                self._field(box, label, var, row=row, column=index * 2, wraplength=150, pady=1)
        ttk.Label(box, text="质量警告", style="Offline.TLabel").grid(row=4, column=0, sticky="nw", padx=(6, 4), pady=(2, 4))
        self.quality_warning_label = ttk.Label(
            box,
            textvariable=self.warnings_var,
            style="OfflineValue.TLabel",
            wraplength=520,
            justify="left",
            anchor="nw",
        )
        self.quality_warning_label.grid(row=4, column=1, columnspan=5, sticky="ew", padx=(0, 6), pady=(2, 4))
        box.bind("<Configure>", lambda event: PostureOfflineSerialApp._update_quality_warning_wraplength(self, event), add="+")

    def _update_quality_warning_wraplength(self, event: object | None = None) -> None:
        label = getattr(self, "quality_warning_label", None)
        if label is None:
            return
        width = int(getattr(event, "width", 0) or 0)
        if width <= 0:
            try:
                width = int(label.master.winfo_width())
            except Exception:
                width = 640
        label.configure(wraplength=max(320, width - 150))

    def _build_tables(self, parent: ttk.Frame) -> None:
        self.tables_container = parent
        parent.rowconfigure(3, weight=0, minsize=RIGHT_NOTEBOOK_MIN_HEIGHT)
        parent.columnconfigure(0, weight=1)
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(4, 4))
        toolbar.columnconfigure(0, weight=1)
        ttk.Label(toolbar, text="姿势统计 / 姿势片段", style="Offline.TLabel").grid(row=0, column=0, sticky="w")
        self.expand_stats_button = ttk.Button(toolbar, text="展开统计", command=self.expand_statistics_window, style="Offline.TButton")
        self.expand_stats_button.grid(row=0, column=1, sticky="e")

        notebook = ttk.Notebook(parent, height=RIGHT_NOTEBOOK_MIN_HEIGHT)
        notebook.grid(row=3, column=0, sticky="nsew")
        self.notebook = notebook
        stats_frame = ttk.Frame(notebook)
        seg_frame = ttk.Frame(notebook)
        self.stats_tab = stats_frame
        self.segment_tab = seg_frame
        notebook.add(stats_frame, text="姿势统计")
        notebook.add(seg_frame, text="姿势片段")
        self.stats_table, self.stats_v_scrollbar, self.stats_h_scrollbar = self._build_tree_table(
            stats_frame,
            [
                ("posture", "姿势", 120),
                ("frames", "帧数", 46),
                ("duration", "持续时间", 64),
                ("share", "占比", 54),
                ("confidence", "平均置信度", 70),
                ("segments", "segment数量", 60),
            ],
            height=9,
        )
        self.segment_table, self.segment_v_scrollbar, self.segment_h_scrollbar = self._build_tree_table(
            seg_frame,
            [
                ("index", "序号", 42),
                ("start", "开始", 46),
                ("end", "结束", 46),
                ("duration", "持续", 52),
                ("type", "类型", 78),
                ("posture", "姿势", 78),
                ("confidence", "平均置信度", 62),
                ("boundary", "Boundary帧", 58),
            ],
            height=9,
        )
        self.segment_table.bind("<<TreeviewSelect>>", self._on_segment_selected)

    def _build_tree_table(
        self,
        parent: ttk.Frame,
        specs: list[tuple[str, str, int]],
        *,
        height: int,
    ) -> tuple[ttk.Treeview, ttk.Scrollbar, ttk.Scrollbar]:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        columns = tuple(item[0] for item in specs)
        tree = ttk.Treeview(parent, columns=columns, show="headings", style="Offline.Treeview", height=height)
        y_scroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        for col, text, width in specs:
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor="w", stretch=True)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self._bind_treeview_mousewheel(tree)
        return tree, y_scroll, x_scroll

    def _bind_treeview_mousewheel(self, tree: ttk.Treeview) -> None:
        tree.bind("<MouseWheel>", lambda event, target=tree: self._scroll_treeview_for_mousewheel(target, event))
        tree.bind("<Button-4>", lambda event, target=tree: self._scroll_treeview_for_mousewheel(target, event))
        tree.bind("<Button-5>", lambda event, target=tree: self._scroll_treeview_for_mousewheel(target, event))

    def _scroll_treeview_for_mousewheel(self, tree: ttk.Treeview, event: object) -> str | None:
        units = _mousewheel_scroll_units(event)
        if not units:
            return "break"
        if _treeview_can_scroll(tree, units):
            tree.yview_scroll(units, "units")
            return "break"
        if _is_descendant(tree, getattr(self, "right_scroll_container", None)):
            canvas = getattr(self, "right_scroll_canvas", None)
            if canvas is not None:
                canvas.yview_scroll(units, "units")
                return "break"
        return None

    def _field(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        *,
        row: int,
        column: int,
        columnspan: int = 1,
        wraplength: int = 260,
        pady: int | tuple[int, int] = 2,
    ) -> ttk.Label:
        ttk.Label(parent, text=label, style="Offline.TLabel").grid(row=row, column=column, sticky="w", padx=(6, 4), pady=pady)
        value_label = ttk.Label(parent, textvariable=variable, style="OfflineValue.TLabel", wraplength=wraplength, justify="left", anchor="w")
        value_label.grid(
            row=row,
            column=column + 1,
            columnspan=columnspan,
            sticky="ew",
            padx=(0, 8),
            pady=pady,
        )
        return value_label

    def choose_capture_folder(self) -> None:
        selected = filedialog.askdirectory(title="选择采集文件夹")
        if selected:
            self._set_selected_path(Path(selected))

    def choose_serial_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择串口数据",
            filetypes=[("Serial data", "*.bin *.txt"), ("Binary", "*.bin"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if selected:
            self._set_selected_path(Path(selected))

    def _set_selected_path(self, path: Path) -> None:
        try:
            selection = select_serial_input(path)
        except Exception as exc:
            messagebox.showerror("无法选择输入", str(exc))
            return
        self.selected_path = path
        self.path_var.set(str(selection.input_path))
        self.input_file_var.set(selection.input_path.name)
        self.input_directory_var.set(str(selection.input_path.parent))
        self.input_type_var.set(selection.input_type)
        self.file_size_var.set(f"{selection.input_path.stat().st_size} bytes")
        self.metadata_path_var.set("—" if selection.metadata_path is None else str(selection.metadata_path))
        self.manual_label_var.set(_display_text(selection.metadata.get("label")))
        self.trial_var.set(_display_text(selection.metadata.get("trial")))
        self.manual_label_trial_var.set(format_manual_label(selection.metadata.get("label"), selection.metadata.get("trial")))
        self.capture_time_var.set(format_capture_time(selection.metadata.get("start_time")))
        orientation, orientation_warnings = resolve_orientation(selection.metadata, fallback_orientation=self.orientation_var.get())
        self.orientation_var.set(format_direction(orientation))
        fps, fps_source, fps_warnings = resolve_fps(selection.metadata, manual_fps=None)
        self.fps_var.set(f"{fps:.4g}")
        self.fps_source_var.set(fps_source)
        self.valid_packets_var.set("—")
        self.invalid_packets_var.set("—")
        self.data_complete_var.set(format_integrity_status(selection.metadata.get("capture_completed")))
        self.calibration_var.set("—")
        self.warnings_var.set(_format_warnings_for_display(selection.warnings + orientation_warnings + fps_warnings))
        self._set_idle_controls()
        self.progress_var.set("已选择输入，等待开始分析。")

    def start_analysis(self) -> None:
        if self.selected_path is None:
            messagebox.showwarning("未选择文件", "请先选择采集文件夹或串口数据。")
            return
        try:
            fps = float(self.fps_var.get())
            if fps <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("FPS无效", "FPS必须是大于0的数字。")
            return
        self.cancel_event.clear()
        self.analysis_result = None
        self._set_analyzing_controls()
        self.progress_var.set("正在解析串口数据并运行V2.4.3识别...")
        path = self.selected_path
        orientation = self.orientation_var.get()
        self.worker_thread = threading.Thread(
            target=self._analysis_worker,
            args=(path, orientation, fps),
            name="OfflineSerialAnalysis",
            daemon=True,
        )
        self.worker_thread.start()

    def _analysis_worker(self, path: Path, orientation: str, fps: float) -> None:
        try:
            parse_result = parse_serial_input(path)
            analyzer = OfflinePostureAnalyzer(model_version=self.model_version)

            def progress(current: int, total: int) -> None:
                self.event_queue.put(("progress", (current, total)))

            result = analyzer.analyze(
                parse_result,
                orientation=orientation,
                fps=fps,
                progress_callback=progress,
                cancel_event=self.cancel_event,
            )
            if result.summary.result_status == "ANALYSIS_CANCELLED":
                self.event_queue.put(("cancelled", result))
            else:
                self.event_queue.put(("done", result))
        except Exception as exc:
            self.event_queue.put(("error", exc))

    def cancel_analysis(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.progress_var.set("当前没有正在进行的分析。")
            return
        self.cancel_event.set()
        self.progress_var.set("正在取消分析...")

    def export_report(self) -> None:
        if self.analysis_result is None:
            messagebox.showwarning("没有分析结果", "请先完成一次离线分析。")
            return
        selected = filedialog.askdirectory(title="选择报告导出目录")
        if not selected:
            return
        try:
            outputs = export_offline_analysis(self.analysis_result, selected)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        messagebox.showinfo("导出完成", f"报告已导出到：\n{outputs['directory']}")

    def show_model_details(self) -> None:
        try:
            recognizer = load_runtime_recognizer(model_version=self.model_version)
            info = model_export_info(recognizer)
        except Exception as exc:
            messagebox.showerror("模型详情", f"模型信息读取失败：{exc}")
            return
        win = tk.Toplevel(self.root)
        win.title("模型详情")
        win.geometry("760x520")
        text = scrolledtext.ScrolledText(win, wrap="word", font=(OFFLINE_FONT_FAMILY, 11))
        text.pack(fill="both", expand=True)
        text.insert("1.0", json.dumps(info, ensure_ascii=False, indent=2))
        text.configure(state="disabled")

    def expand_statistics_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("姿势统计 / 姿势片段")
        win.geometry("900x600")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(win)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        stats_frame = ttk.Frame(notebook)
        segment_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="姿势统计")
        notebook.add(segment_frame, text="姿势片段")
        stats_table, _stats_y, _stats_x = self._build_tree_table(
            stats_frame,
            [
                ("posture", "姿势", 180),
                ("frames", "帧数", 90),
                ("duration", "持续时间", 110),
                ("share", "占比", 90),
                ("confidence", "平均置信度", 120),
                ("segments", "segment数量", 120),
            ],
            height=14,
        )
        segment_table, _seg_y, _seg_x = self._build_tree_table(
            segment_frame,
            [
                ("index", "序号", 60),
                ("start", "开始", 90),
                ("end", "结束", 90),
                ("duration", "持续", 90),
                ("type", "类型", 150),
                ("posture", "姿势", 150),
                ("confidence", "平均置信度", 120),
                ("boundary", "Boundary帧", 110),
            ],
            height=14,
        )
        if self.analysis_result is not None:
            self._populate_stats_table(stats_table, self.analysis_result)
            self._populate_segment_table(segment_table, self.analysis_result)

            def on_popup_segment(_event: object) -> None:
                selection = segment_table.selection()
                if not selection or self.analysis_result is None:
                    return
                index = int(selection[0])
                segment = self.analysis_result.posture_segments[index]
                self.current_frame_index = segment.start_frame
                self._render_current_frame()

            segment_table.bind("<<TreeviewSelect>>", on_popup_segment)

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "progress":
                    current, total = payload
                    self.progress_var.set(f"正在分析：{current}/{total} 帧")
                elif kind == "done":
                    self._handle_analysis_done(payload)
                elif kind == "cancelled":
                    self._handle_analysis_cancelled(payload)
                elif kind == "error":
                    self._handle_analysis_error(payload)
        except Empty:
            pass
        self.after_id = self.root.after(100, self._poll_worker)

    def _handle_analysis_done(self, result: OfflineAnalysisResult) -> None:
        self.analysis_result = result
        self.current_frame_index = 0
        self._update_file_and_parse_info(result)
        self._update_summary(result)
        self._update_tables(result)
        self.frame_scale.configure(to=max(0, len(result.frames) - 1))
        self._render_current_frame()
        self._focus_statistics_panel()
        self.progress_var.set(f"分析完成：{len(result.frame_predictions)}帧，综合判定 {result.summary.overall_posture or '无法可靠确定'}。")
        self._set_idle_controls()

    def _handle_analysis_cancelled(self, result: OfflineAnalysisResult) -> None:
        self.analysis_result = result
        self.progress_var.set("分析已取消。")
        self._set_idle_controls()

    def _handle_analysis_error(self, exc: BaseException) -> None:
        self.progress_var.set(f"分析失败：{exc}")
        messagebox.showerror("分析失败", str(exc))
        self._set_idle_controls()

    def _update_file_and_parse_info(self, result: OfflineAnalysisResult) -> None:
        stats = result.parser_stats
        input_path = Path(result.input_path)
        self.path_var.set(str(input_path))
        self.input_file_var.set(input_path.name)
        self.input_directory_var.set(str(input_path.parent))
        self.input_type_var.set(result.input_type)
        try:
            self.file_size_var.set(f"{input_path.stat().st_size} bytes")
        except OSError:
            self.file_size_var.set("—")
        self.fps_var.set(f"{result.fps:.4g}")
        self.fps_source_var.set(result.fps_source)
        self.orientation_var.set(format_direction(result.orientation))
        self.manual_label_var.set(_display_text(result.metadata.get("label")))
        self.trial_var.set(_display_text(result.metadata.get("trial")))
        self.manual_label_trial_var.set(format_manual_label(result.metadata.get("label"), result.metadata.get("trial")))
        self.capture_time_var.set(format_capture_time(result.metadata.get("start_time")))
        self.total_bytes_var.set(str(stats.get("total_bytes", "—")))
        self.valid_packets_var.set(str(stats.get("valid_packets", "—")))
        self.invalid_packets_var.set(str(stats.get("invalid_packets", "—")))
        self.discarded_bytes_var.set(str(stats.get("discarded_bytes", "—")))
        self.invalid_lines_var.set(str(stats.get("invalid_text_line_count", "—")))
        self.data_complete_var.set(_data_complete_status(result.metadata, stats, result.invalid_text_lines))
        self.calibration_var.set(format_calibration_status(result.calibration_info.calibration_status))

    def _update_summary(self, result: OfflineAnalysisResult) -> None:
        summary = result.summary
        self.overall_var.set(summary.overall_posture or "无法可靠确定")
        self.status_var.set(summary.result_status)
        self.dominant_var.set(summary.dominant_posture or "—")
        self.share_var.set(_format_percent(summary.dominant_posture_share))
        self.mean_conf_var.set(_format_optional(summary.mean_confidence))
        self.boundary_rate_var.set(_format_percent(summary.boundary_rate))
        self.human_duration_var.set(f"{summary.human_duration_s:.2f}s")
        self.stable_duration_var.set(f"{summary.stable_posture_duration_s:.2f}s")
        if summary.label_matches_overall is None:
            self.label_match_var.set("—")
        else:
            self.label_match_var.set("是" if summary.label_matches_overall else "否")
        self.warnings_var.set(_format_warnings_for_display(_warnings_for_result_display(result)))

    def _update_tables(self, result: OfflineAnalysisResult) -> None:
        for table in (self.stats_table, self.segment_table):
            for item in table.get_children():
                table.delete(item)
        self._populate_stats_table(self.stats_table, result)
        self._populate_segment_table(self.segment_table, result)

    def _populate_stats_table(self, table: ttk.Treeview, result: OfflineAnalysisResult) -> None:
        for row in posture_statistics(result):
            table.insert(
                "",
                "end",
                values=(
                    row["posture"],
                    row["frame_count"],
                    f"{row['duration_s']:.2f}s",
                    "—" if row["share"] is None else _format_percent(row["share"]),
                    _format_optional(row["mean_confidence"]),
                    row["segment_count"],
                ),
            )

    def _populate_segment_table(self, table: ttk.Treeview, result: OfflineAnalysisResult) -> None:
        for segment in result.posture_segments:
            iid = str(segment.segment_index)
            table.insert(
                "",
                "end",
                iid=iid,
                values=(
                    segment.segment_index,
                    f"{segment.start_time_s:.2f}",
                    f"{segment.end_time_s:.2f}",
                    f"{segment.duration_s:.2f}s",
                    segment.segment_type,
                    segment.posture or "—",
                    _format_optional(segment.mean_confidence),
                    segment.boundary_frame_count,
                ),
            )

    def _focus_statistics_panel(self) -> None:
        if hasattr(self, "notebook") and hasattr(self, "stats_tab"):
            try:
                self.notebook.select(self.stats_tab)
            except Exception:
                pass
        if hasattr(self, "stats_table"):
            try:
                self.stats_table.yview_moveto(0)
            except Exception:
                pass

    def window_layout_diagnostics(self) -> dict[str, Any]:
        try:
            minimum_width, minimum_height = self.root.minsize()
        except Exception:
            minimum_width, minimum_height = MIN_WINDOW_SIZE
        try:
            resizable_width, resizable_height = self.root.resizable()
        except Exception:
            resizable_width, resizable_height = (True, True)
        return {
            "requested_geometry": getattr(self, "requested_geometry", ""),
            "actual_geometry": self.root.geometry(),
            "screen_width": self.root.winfo_screenwidth(),
            "screen_height": self.root.winfo_screenheight(),
            "minimum_width": minimum_width,
            "minimum_height": minimum_height,
            "resizable_width": bool(resizable_width),
            "resizable_height": bool(resizable_height),
            "right_scroll_canvas_height": self.right_scroll_canvas.winfo_height() if hasattr(self, "right_scroll_canvas") else None,
            "right_scroll_content_height": self.right_scroll_content.winfo_reqheight() if hasattr(self, "right_scroll_content") else None,
        }

    def _render_current_frame(self) -> None:
        if self.analysis_result is None or not self.analysis_result.frames:
            self._draw_empty_heatmap()
            return
        index = max(0, min(self.current_frame_index, len(self.analysis_result.frames) - 1))
        self.current_frame_index = index
        frame = self.analysis_result.frames[index]
        prediction = self.analysis_result.frame_predictions[index] if index < len(self.analysis_result.frame_predictions) else None
        self._draw_heatmap(frame)
        self.frame_scale.set(index)
        self.frame_info_var.set(f"当前帧：{index + 1}/{len(self.analysis_result.frames)}")
        if prediction is not None:
            self.frame_state_var.set(f"状态：{prediction.display_status}")
            self.frame_posture_var.set(f"姿势：{prediction.posture or '—'}")
            self.frame_confidence_var.set(f"置信度：{_format_optional(prediction.posture_confidence)}")
            self.frame_boundary_var.set(f"Boundary：{'是' if prediction.is_boundary else '否'}")
        self.total_pressure_var.set(f"总压力：{float(frame.sum()):.1f}")
        self.max_pressure_var.set(f"最大压力：{float(frame.max()) if frame.size else 0.0:.1f}")
        self.active_points_var.set(f"活跃点：{int((frame > 15.0).sum())}")

    def _draw_empty_heatmap(self) -> None:
        self._draw_heatmap(np.zeros((16, 16), dtype=np.float32))

    def _draw_heatmap(self, frame: np.ndarray) -> None:
        width = max(float(self.canvas.winfo_width()), 1.0)
        height = max(float(self.canvas.winfo_height()), 1.0)
        geometry = heatmap_grid_geometry(width, height)
        self.canvas.delete("all")
        self.canvas.configure(bg="#f4f6f8")
        max_value = float(np.asarray(frame).max()) if np.asarray(frame).size else 0.0
        for row in range(16):
            for col in range(16):
                x0, y0, x1, y1 = geometry.bounds_for(row, col)
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=pressure_to_color(float(frame[row, col]), max_value),
                    outline="#e2e8f0",
                    width=1,
                )
        self.canvas.create_rectangle(
            geometry.offset_x,
            geometry.offset_y,
            geometry.offset_x + geometry.square_size,
            geometry.offset_y + geometry.square_size,
            outline="#334155",
            width=2,
        )

    def _on_frame_scale(self, value: str) -> None:
        if self.analysis_result is None:
            return
        target = int(round(float(value)))
        if target != self.current_frame_index:
            self.current_frame_index = target
            self._render_current_frame()

    def previous_frame(self) -> None:
        self.current_frame_index -= 1
        self._render_current_frame()

    def next_frame(self) -> None:
        self.current_frame_index += 1
        self._render_current_frame()

    def jump_start(self) -> None:
        self.current_frame_index = 0
        self._render_current_frame()

    def jump_end(self) -> None:
        if self.analysis_result is not None:
            self.current_frame_index = len(self.analysis_result.frames) - 1
        self._render_current_frame()

    def play(self) -> None:
        if self.analysis_result is None:
            return
        self.playing = True
        self._play_tick()

    def pause(self) -> None:
        self.playing = False
        if self.play_after_id:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def _play_tick(self) -> None:
        if not self.playing or self.analysis_result is None:
            return
        self.next_frame()
        if self.current_frame_index >= len(self.analysis_result.frames) - 1:
            self.pause()
            return
        delay = _playback_delay_ms(self.speed_var.get(), self.analysis_result.fps)
        self.play_after_id = self.root.after(delay, self._play_tick)

    def _on_segment_selected(self, _event: object) -> None:
        selection = self.segment_table.selection()
        if not selection or self.analysis_result is None:
            return
        index = int(selection[0])
        segment = self.analysis_result.posture_segments[index]
        self.current_frame_index = segment.start_frame
        self._render_current_frame()

    def _set_idle_controls(self) -> None:
        has_input = self.selected_path is not None
        has_result = self.analysis_result is not None
        self.folder_button.configure(state="normal")
        self.file_button.configure(state="normal")
        self.orientation_combo.configure(state="readonly")
        self.fps_entry.configure(state="normal")
        self.start_button.configure(state="normal" if has_input else "disabled")
        self.cancel_button.configure(state="disabled")
        self.export_button.configure(state="normal" if has_result else "disabled")

    def _set_analyzing_controls(self) -> None:
        self.folder_button.configure(state="disabled")
        self.file_button.configure(state="disabled")
        self.orientation_combo.configure(state="disabled")
        self.fps_entry.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.export_button.configure(state="disabled")

    def _on_close(self) -> None:
        self.cancel_event.set()
        self.pause()
        if self.after_id:
            self.root.after_cancel(self.after_id)
        PostureOfflineSerialApp._unbind_global_mousewheel(self)
        self.root.destroy()


def _format_optional(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def _format_percent(value: float | None) -> str:
    return "—" if value is None else f"{float(value) * 100.0:.1f}%"


def _display_text(value: object) -> str:
    if value is None:
        return "—"
    text = str(value).strip()
    return text if text else "—"


def format_capture_time(value: object) -> str:
    text = _display_text(value)
    if text == "—":
        return text
    normalized = text.replace("/", "-").replace("_", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    try:
        date_part, time_part = normalized.split("T", 1) if "T" in normalized else normalized.split(" ", 1)
        year, month, day = [int(part) for part in date_part.split("-")[:3]]
        hour, minute, second = time_part.split(":")[:3]
        second = second.split(".", 1)[0]
        return f"{year:04d}-{month:02d}-{day:02d} {int(hour):02d}:{int(minute):02d}:{int(second):02d}"
    except Exception:
        return text


def format_manual_label(label: object, trial: object) -> str:
    label_text = _display_text(label)
    trial_text = _display_text(trial)
    if label_text == "—" and trial_text == "—":
        return "—"
    if trial_text == "—":
        return label_text
    try:
        number = int(float(trial_text))
        trial_display = f"第{number}次"
    except ValueError:
        trial_display = f"第{trial_text}次"
    return trial_display if label_text == "—" else f"{label_text} · {trial_display}"


def format_direction(value: object) -> str:
    text = _display_text(value)
    return "未记录" if text == "—" else text


def format_integrity_status(value: object) -> str:
    if isinstance(value, bool):
        return "完整" if value else "不完整"
    text = _display_text(value)
    if text == "—":
        return text
    lowered = text.lower()
    if lowered == "true":
        return "完整"
    if lowered == "false":
        return "不完整"
    return text


def format_calibration_status(value: object) -> str:
    mapping = {
        "NO_RELIABLE_EMPTY_BASELINE": "未找到可靠空载基线",
        "CALIBRATED_FROM_INITIAL_EMPTY": "已使用开头空载数据校准",
        "CALIBRATION_SKIPPED": "未执行空载校准",
        "CALIBRATION_FAILED": "空载校准失败",
    }
    text = _display_text(value)
    return mapping.get(text, text)


def _data_complete_status(metadata: dict[str, Any], stats: dict[str, Any], invalid_text_lines: Sequence[object]) -> str:
    if "capture_completed" in metadata:
        return format_integrity_status(metadata.get("capture_completed"))
    has_warning = (
        _stat_int(stats, "invalid_packets") > 0
        or _stat_int(stats, "trailing_incomplete_bytes") > 0
        or _stat_int(stats, "invalid_text_line_count") > 0
        or bool(invalid_text_lines)
    )
    return "不完整" if has_warning else "完整"


def _format_warnings_for_display(warnings: Sequence[str] | str | None) -> str:
    if warnings is None:
        return "无"
    if isinstance(warnings, str):
        text = warnings.strip()
        return text if text and text != "—" else "无"
    lines = [str(warning).strip() for warning in warnings if str(warning).strip()]
    return "\n".join(lines) if lines else "无"


def _warnings_for_result_display(result: object) -> list[str]:
    summary = getattr(result, "summary", None)
    warnings = list(getattr(summary, "warnings", []) or [])
    stats = getattr(result, "parser_stats", {}) or {}

    def add_warning(text: str) -> None:
        if text not in warnings:
            warnings.append(text)

    invalid_packets = _stat_int(stats, "invalid_packets")
    if invalid_packets > 0:
        add_warning(f"存在{invalid_packets}个无效协议包")
    discarded_bytes = _stat_int(stats, "discarded_bytes")
    if discarded_bytes > 0:
        add_warning(f"丢弃了{discarded_bytes}个非协议字节")
    invalid_text_lines = _stat_int(stats, "invalid_text_line_count")
    if invalid_text_lines > 0:
        add_warning(f"串口文本中存在{invalid_text_lines}个无效行")
    trailing_bytes = _stat_int(stats, "trailing_incomplete_bytes")
    if trailing_bytes > 0 and not any("文件末尾存在未完成" in warning for warning in warnings):
        add_warning("文件末尾存在未完成的数据包")
    return warnings


def _stat_int(stats: dict[str, Any], key: str) -> int:
    try:
        return int(stats.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _is_descendant(widget: object | None, ancestor: object | None) -> bool:
    current = widget
    while current is not None and ancestor is not None:
        if current is ancestor:
            return True
        current = getattr(current, "master", None)
    return False


def _treeview_can_scroll(tree: ttk.Treeview, units: int) -> bool:
    try:
        first, last = tree.yview()
        first = float(first)
        last = float(last)
    except Exception:
        return True
    if units < 0:
        return first > 0.0
    if units > 0:
        return last < 1.0
    return False


def _playback_delay_ms(speed: str, fps: float) -> int:
    if speed == "最大速度":
        return 1
    multiplier = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "5×": 5.0}.get(speed, 1.0)
    return max(1, int(round(1000.0 / max(fps * multiplier, 1e-6))))


def main(
    argv: Sequence[str] | None = None,
    *,
    brand_name: str | None = None,
    subtitle: str | None = None,
    app_title: str | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> int:
    parser = argparse.ArgumentParser(description="Offline serial posture analysis desktop app.")
    parser.add_argument("--model-version", default=model_version, help="Recognizer model version to load.")
    args = parser.parse_args(argv)
    root = tk.Tk()
    PostureOfflineSerialApp(
        root,
        model_version=args.model_version,
        brand_name=brand_name,
        subtitle=subtitle,
        app_title=app_title,
    )
    root.mainloop()
    return 0
