from __future__ import annotations

import argparse
import json
from queue import Empty, Queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
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


DEFAULT_MODEL_VERSION = "v2_4_3_candidate"


class PostureSerialApp:
    def __init__(self, root: tk.Tk, model_version: str = DEFAULT_MODEL_VERSION) -> None:
        self.root = root
        self.model_version = model_version
        self.root.title(f"实时串口坐姿识别 - 当前模型：{model_version_display_name(self.model_version)}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.minsize(1040, 680)

        self.recognizer: object | None = None
        self.reader: SerialFrameReader | None = None
        self.worker: RecognitionWorker | None = None
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

        self.state_var = tk.StringVar(value="空载")
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
        self.active_var = tk.StringVar(value="0")

        self._build_ui()
        self._draw_heatmap(np.zeros((16, 16), dtype=np.float32))
        self.refresh_ports()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(toolbar, text="串口").grid(row=0, column=0, padx=(0, 4))
        self.port_combo = ttk.Combobox(toolbar, textvariable=self.port_var, width=28, state="readonly")
        self.port_combo.grid(row=0, column=1, padx=3)
        ttk.Button(toolbar, text="刷新串口", command=self.refresh_ports).grid(row=0, column=2, padx=3)
        ttk.Button(toolbar, text="连接", command=self.connect).grid(row=0, column=3, padx=3)
        ttk.Button(toolbar, text="断开", command=self.disconnect).grid(row=0, column=4, padx=3)
        ttk.Button(toolbar, text="空载校准", command=self.calibrate_empty).grid(row=0, column=5, padx=3)
        ttk.Button(toolbar, text="重置识别状态", command=self.reset_recognition_state).grid(row=0, column=6, padx=3)
        ttk.Button(toolbar, text="查看模型详情", command=self.show_model_details).grid(row=0, column=7, padx=3)
        ttk.Label(toolbar, text="方向").grid(row=0, column=8, padx=(12, 4))
        self.orientation_combo = ttk.Combobox(toolbar, textvariable=self.orientation_var, values=list(ORIENTATION_MODES), width=18, state="readonly")
        self.orientation_combo.grid(row=0, column=9, padx=3)
        self.orientation_combo.bind("<<ComboboxSelected>>", self._on_orientation_changed)

        left = ttk.Frame(self.root, padding=(12, 8))
        left.grid(row=1, column=0, sticky="nsew")
        left.columnconfigure(1, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="前", anchor="center").grid(row=0, column=1, sticky="ew")
        ttk.Label(left, text="左", anchor="center").grid(row=1, column=0, sticky="ns")
        canvas_bg = str(self.root.cget("bg"))
        self.canvas = tk.Canvas(left, width=480, height=480, bg=canvas_bg, highlightthickness=0)
        self.canvas.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda _event: self._draw_heatmap(self.current_frame))
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
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        self._build_status_panel(right)

        bottom = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(bottom, textvariable=self.summary_var).grid(row=0, column=0, sticky="w")

    def _build_status_panel(self, parent: ttk.Frame) -> None:
        connection_box = ttk.LabelFrame(parent, text="连接信息", padding=8)
        connection_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._field(connection_box, "连接状态", self.connection_var, 0)
        self._field(connection_box, "当前端口", self.current_port_var, 1)
        self._field(connection_box, "波特率", self.baud_var, 2)
        self._field(connection_box, "接收字节数", self.received_bytes_var, 3)
        self._field(connection_box, "有效协议帧", self.valid_frames_var, 4)
        self._field(connection_box, "无效协议帧", self.invalid_frames_var, 5)
        self._field(connection_box, "丢弃字节数", self.discarded_bytes_var, 6)
        self._field(connection_box, "接收FPS", self.receive_fps_var, 7)
        self._field(connection_box, "frame queue长度", self.frame_queue_var, 8)
        self._field(connection_box, "丢弃queue帧", self.dropped_queue_var, 9)
        self._field(connection_box, "最近串口错误", self.serial_error_var, 10)

        result_box = ttk.LabelFrame(parent, text="识别信息", padding=8)
        result_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._field(result_box, "当前系统状态", self.state_var, 0)
        self._field(result_box, "Occupancy", self.occupancy_var, 1)
        self._field(result_box, "姿势", self.posture_var, 2)
        self._field(result_box, "原始候选", self.raw_var, 3)
        self._field(result_box, "confidence", self.confidence_var, 4)
        self._field(result_box, "second_label", self.second_var, 5)
        self._field(result_box, "margin", self.margin_var, 6)
        self._field(result_box, "Boundary", self.boundary_var, 7)
        self._field(result_box, "Boundary原因", self.boundary_reason_var, 8)
        self._field(result_box, "Prototype", self.prototype_var, 9)
        self._field(result_box, "当前帧序号", self.frame_index_var, 10)
        self._field(result_box, "连接运行时间", self.uptime_var, 11)
        self._field(result_box, "最近推理耗时", self.inference_var, 12)
        self._field(result_box, "平均推理耗时", self.average_inference_var, 13)
        self._field(result_box, "最近识别错误", self.recognition_error_var, 14)

    def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=variable, width=34).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)

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
        self._schedule_poll()

    def _finish_connect_error(self, title: str, message: str) -> None:
        self.connecting = False
        self.connection_var.set("未连接")
        self.summary_var.set(message)
        messagebox.showerror(title, message)

    def _on_orientation_changed(self, _event: tk.Event | None = None) -> None:
        self.orientation_state.set(self.orientation_var.get())

    def disconnect(self) -> None:
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
        self.summary_var.set("已断开。")

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
        self.active_var.set(str(record.active_points))
        self.state_var.set(STATE_LABELS.get(record.display_status, record.display_status))
        self.occupancy_var.set(record.occupancy_state)
        self.posture_var.set(record.posture or ("边界姿势/低置信度" if record.display_status == "POSTURE" and record.is_boundary else "—"))
        self.raw_var.set(_format_raw_candidate(record))
        self.confidence_var.set(_format_optional(record.posture_confidence))
        self.second_var.set(record.second_label or "—")
        self.margin_var.set(_format_optional(record.margin))
        self.boundary_var.set("是" if record.is_boundary else "否")
        self.boundary_reason_var.set(record.boundary_reason or record.lateral_boundary_reasons or "—")
        self.prototype_var.set(_prototype_label(record))
        self.frame_index_var.set(str(record.frame_index))
        self.uptime_var.set(f"{record.timestamp:.2f}s")
        self.inference_var.set(f"{result.inference_ms:.1f} ms")
        if self.worker and self.worker.average_inference_ms is not None:
            self.average_inference_var.set(f"{self.worker.average_inference_ms:.1f} ms")

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

    def _report_background_errors(self) -> None:
        if self.reader is not None and self.reader.last_error is not None:
            message = str(self.reader.last_error)
            if message != self._reported_reader_error:
                self._reported_reader_error = message
                self.summary_var.set("设备连接已断开或串口读取发生错误。")
                messagebox.showerror("串口错误", "设备连接已断开或串口读取发生错误。")
        if self.worker is not None and self.worker.last_error is not None:
            message = str(self.worker.last_error)
            if message != self._reported_worker_error:
                self._reported_worker_error = message
                self.summary_var.set(f"识别线程发生错误：{message}")
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

    def _call_recognizer_reset(self) -> None:
        reset = getattr(self.recognizer, "reset", None)
        if callable(reset):
            reset()

    def _on_close(self) -> None:
        self.disconnect()
        self.root.destroy()


def _recommended_port(devices: list[str]) -> str | None:
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


def main(argv: Sequence[str] | None = None) -> int:
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
    PostureSerialApp(root, model_version=args.model_version)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
