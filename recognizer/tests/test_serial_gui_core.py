from __future__ import annotations

import importlib
import os
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from recognizer.frame_orientation import apply_sensor_rotation
from recognizer.data_loader import read_sensor_csv
from recognizer.serial_gui_core import ORIENTATION_MODES, RecognitionWorker, apply_orientation, apply_sensor_and_orientation
from recognizer.serial_protocol import ParsedPressureFrame
from recognizer.serial_recorder import SerialDataRecorder


class FakeTkWidget:
    instances: list["FakeTkWidget"] = []

    def __init__(self, master: object | None = None, **kwargs: object) -> None:
        self.master = master
        self.kwargs = kwargs
        self.grid_kwargs: dict[str, object] | None = None
        self.rowconfigure_calls: list[tuple[int, dict[str, object]]] = []
        self.columnconfigure_calls: list[tuple[int, dict[str, object]]] = []
        FakeTkWidget.instances.append(self)

    def grid(self, **kwargs: object) -> None:
        self.grid_kwargs = kwargs

    def rowconfigure(self, index: int, **kwargs: object) -> None:
        self.rowconfigure_calls.append((index, kwargs))

    def columnconfigure(self, index: int, **kwargs: object) -> None:
        self.columnconfigure_calls.append((index, kwargs))

    def configure(self, **_kwargs: object) -> None:
        self.kwargs.update(_kwargs)

    def bind(self, *_args: object, **_kwargs: object) -> None:
        return


class FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


def fake_widget_factory(**extra_kwargs: object):
    def make_widget(master: object | None = None, **kwargs: object) -> FakeTkWidget:
        return FakeTkWidget(master, **{**extra_kwargs, **kwargs})

    return make_widget


def wait_until(predicate, *, timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def posture_payload(label: str = "端正坐姿") -> dict[str, object]:
    return {
        "occupancy": "HUMAN",
        "seat_state": "HUMAN",
        "posture": label,
        "posture_confidence": 0.91,
        "raw_label": label,
        "raw_confidence": 0.91,
        "second_label": "前倾端坐",
        "margin": 0.42,
        "is_boundary": False,
        "occupancy_features": {"total_pressure": 123.0, "active_points": 8},
    }


def asymmetric_corner_frame() -> np.ndarray:
    frame = np.zeros((16, 16), dtype=np.float32)
    frame[0, 0] = 11.0
    frame[0, 15] = 22.0
    frame[15, 0] = 33.0
    frame[15, 15] = 44.0
    return frame


class FakeRecognizer:
    def __init__(self, *, fail_on_predict: bool = False, sleep_s: float = 0.0) -> None:
        self.fail_on_predict = fail_on_predict
        self.sleep_s = sleep_s
        self.predict_frames: list[np.ndarray] = []
        self.reset_count = 0
        self.calibrate_frames: list[np.ndarray] = []
        self.in_predict = False
        self.concurrent_access = False

    def predict(self, frame: np.ndarray) -> dict[str, object]:
        if self.in_predict:
            self.concurrent_access = True
        self.in_predict = True
        try:
            self.predict_frames.append(np.array(frame, copy=True))
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self.fail_on_predict:
                raise RuntimeError("predict exploded")
            return posture_payload(label=f"姿势{len(self.predict_frames)}")
        finally:
            self.in_predict = False

    def reset(self) -> None:
        if self.in_predict:
            self.concurrent_access = True
        self.reset_count += 1

    def calibrate(self, frame: np.ndarray) -> dict[str, object]:
        if self.in_predict:
            self.concurrent_access = True
        self.calibrate_frames.append(np.array(frame, copy=True))
        return {"calibrated": True}


class StepClock:
    def __init__(self, start: float = 100.0, step: float = 0.1) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class SerialGuiCoreOrientationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.arange(256, dtype=np.float32).reshape(16, 16)

    def test_apply_orientation_supports_all_eight_modes(self) -> None:
        expected = {
            "原始": self.frame,
            "上下翻转": np.flipud(self.frame),
            "左右翻转": np.fliplr(self.frame),
            "上下及左右翻转": np.flipud(np.fliplr(self.frame)),
            "转置": self.frame.T,
            "转置后上下翻转": np.flipud(self.frame.T),
            "转置后左右翻转": np.fliplr(self.frame.T),
            "转置后上下及左右翻转": np.flipud(np.fliplr(self.frame.T)),
        }

        self.assertEqual(tuple(ORIENTATION_MODES), tuple(expected))
        for mode, matrix in expected.items():
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(apply_orientation(self.frame, mode), matrix)

    def test_apply_orientation_does_not_modify_input(self) -> None:
        original = self.frame.copy()

        transformed = apply_orientation(self.frame, "左右翻转")
        transformed[0, 0] = -999.0

        np.testing.assert_array_equal(self.frame, original)

    def test_apply_orientation_returns_16x16_contiguous_array(self) -> None:
        transformed = apply_orientation(self.frame, "转置后左右翻转")

        self.assertEqual(transformed.shape, (16, 16))
        self.assertTrue(transformed.flags["C_CONTIGUOUS"])

    def test_sensor_rotation_and_legacy_orientation_combinations_are_explicit(self) -> None:
        frame = asymmetric_corner_frame()

        cases = {
            (0, "原始"): np.array([[11.0, 22.0], [33.0, 44.0]], dtype=np.float32),
            (180, "原始"): np.array([[44.0, 33.0], [22.0, 11.0]], dtype=np.float32),
            (180, "左右翻转"): np.array([[33.0, 44.0], [11.0, 22.0]], dtype=np.float32),
            (180, "上下翻转"): np.array([[22.0, 11.0], [44.0, 33.0]], dtype=np.float32),
            (180, "上下及左右翻转"): np.array([[11.0, 22.0], [33.0, 44.0]], dtype=np.float32),
        }

        for (rotation, orientation), corners in cases.items():
            with self.subTest(rotation=rotation, orientation=orientation):
                transformed = apply_sensor_and_orientation(frame, rotation, orientation)
                observed = np.array(
                    [
                        [transformed[0, 0], transformed[0, 15]],
                        [transformed[15, 0], transformed[15, 15]],
                    ],
                    dtype=np.float32,
                )
                np.testing.assert_array_equal(observed, corners)


class SerialGuiLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTkWidget.instances.clear()

    def make_fake_app(self) -> SimpleNamespace:
        import recognizer.serial_gui as serial_gui

        app = SimpleNamespace(
            root=FakeTkWidget(),
            capture_label_var=object(),
            capture_trial_var=object(),
            capture_dir_var=object(),
            capture_status_var=object(),
            capture_duration_var=object(),
            capture_raw_bytes_var=object(),
            capture_frames_var=object(),
            capture_predictions_var=object(),
            capture_queue_var=object(),
            capture_dropped_var=object(),
            capture_error_var=object(),
            sensor_rotation_var=object(),
            sensor_rotation_status_var=object(),
            connection_var=object(),
            current_port_var=object(),
            baud_var=object(),
            received_bytes_var=object(),
            valid_frames_var=object(),
            invalid_frames_var=object(),
            discarded_bytes_var=object(),
            receive_fps_var=object(),
            frame_queue_var=object(),
            dropped_queue_var=object(),
            serial_error_var=object(),
            state_var=object(),
            summary_status_var=object(),
            summary_posture_var=object(),
            summary_confidence_var=object(),
            summary_boundary_var=object(),
            occupancy_var=object(),
            posture_var=object(),
            raw_var=object(),
            confidence_var=object(),
            second_var=object(),
            margin_var=object(),
            boundary_var=object(),
            boundary_reason_var=object(),
            prototype_var=object(),
            frame_index_var=object(),
            uptime_var=object(),
            inference_var=object(),
            average_inference_var=object(),
            recognition_error_var=object(),
            _wrap_labels=[],
            choose_capture_directory=lambda: None,
            start_capture=lambda: None,
            stop_capture=lambda: None,
        )
        app._capture_field = lambda *args, **kwargs: serial_gui.PostureSerialApp._capture_field(app, *args, **kwargs)
        app._field = lambda *args, **kwargs: serial_gui.PostureSerialApp._field(app, *args, **kwargs)
        app._long_text_field = lambda *args, **kwargs: serial_gui.PostureSerialApp._long_text_field(app, *args, **kwargs)
        app._update_recognition_wraps = lambda *args, **kwargs: serial_gui.PostureSerialApp._update_recognition_wraps(app, *args, **kwargs)
        return app

    def test_capture_panel_uses_three_responsive_rows(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = self.make_fake_app()
        with (
            patch.object(serial_gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Frame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Label", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Entry", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Button", fake_widget_factory()),
        ):
            serial_gui.PostureSerialApp._build_capture_panel(app, row=0)

        capture = next(widget for widget in FakeTkWidget.instances if widget.kwargs.get("text") == "数据采集")
        direct_child_rows = [
            int(widget.grid_kwargs["row"])
            for widget in FakeTkWidget.instances
            if widget.master is capture and widget.grid_kwargs is not None
        ]
        child_texts = {widget.kwargs.get("text") for widget in FakeTkWidget.instances}

        self.assertEqual(set(direct_child_rows), {0, 1, 2})
        for required_label in ("保存目录", "有效帧", "识别结果", "时长", "原始字节", "写入队列", "丢弃事件", "最近写入错误"):
            self.assertIn(required_label, child_texts)
        self.assertIn((0, {"weight": 1}), capture.columnconfigure_calls)

    def test_capture_path_and_error_values_use_bounded_responsive_widths(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = self.make_fake_app()
        with (
            patch.object(serial_gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Frame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Label", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Entry", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Button", fake_widget_factory()),
        ):
            serial_gui.PostureSerialApp._build_capture_panel(app, row=0)

        value_widgets = [widget for widget in FakeTkWidget.instances if widget.kwargs.get("textvariable") in {app.capture_dir_var, app.capture_error_var}]

        self.assertEqual(len(value_widgets), 2)
        for widget in value_widgets:
            self.assertIn("e", widget.grid_kwargs["sticky"])
            self.assertIn("w", widget.grid_kwargs["sticky"])
            self.assertLessEqual(int(widget.kwargs["width"]), 30)

    def test_capture_primary_button_is_disabled_until_connected(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = self.make_fake_app()
        with (
            patch.object(serial_gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Frame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Label", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Entry", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Button", fake_widget_factory()),
        ):
            serial_gui.PostureSerialApp._build_capture_panel(app, row=0)

        start_button = next(widget for widget in FakeTkWidget.instances if widget.kwargs.get("text") == "开始采集")
        stop_button = next(widget for widget in FakeTkWidget.instances if widget.kwargs.get("text") == "停止采集")

        self.assertEqual(start_button.kwargs.get("style"), serial_gui.SERIAL_PRIMARY_BUTTON_STYLE)
        self.assertEqual(start_button.kwargs.get("state"), "disabled")
        self.assertEqual(stop_button.kwargs.get("style"), serial_gui.SERIAL_DANGER_BUTTON_STYLE)

    def test_sensor_rotation_options_are_explicit_and_default_to_legacy_installation(self) -> None:
        import recognizer.serial_gui as serial_gui

        self.assertEqual(serial_gui.SENSOR_ROTATION_OPTIONS[0], "0°（插电口朝前 / 旧安装方式）")
        self.assertEqual(serial_gui.SENSOR_ROTATION_OPTIONS[1], "180°（插电口朝后 / 沙发安装）")
        self.assertEqual(serial_gui.sensor_rotation_degrees_from_label(serial_gui.SENSOR_ROTATION_OPTIONS[0]), 0)
        self.assertEqual(serial_gui.sensor_rotation_degrees_from_label(serial_gui.SENSOR_ROTATION_OPTIONS[1]), 180)

    def test_capture_inputs_disable_sensor_rotation_while_recording(self) -> None:
        import recognizer.serial_gui as serial_gui

        configured: dict[str, str] = {}
        app = SimpleNamespace(
            capture_label_entry=FakeTkWidget(),
            capture_trial_entry=FakeTkWidget(),
            capture_dir_button=FakeTkWidget(),
            refresh_button=FakeTkWidget(),
            connect_button=FakeTkWidget(),
            capture_start_button=FakeTkWidget(),
            port_combo=FakeTkWidget(),
            orientation_combo=FakeTkWidget(),
            sensor_rotation_combo=FakeTkWidget(),
            reader=object(),
            worker=object(),
        )

        serial_gui.PostureSerialApp._set_capture_inputs_enabled(app, False)
        configured["disabled"] = app.sensor_rotation_combo.kwargs["state"]
        serial_gui.PostureSerialApp._set_capture_inputs_enabled(app, True)
        configured["enabled"] = app.sensor_rotation_combo.kwargs["state"]

        self.assertEqual(configured, {"disabled": "disabled", "enabled": "readonly"})

    def test_sensor_rotation_change_clears_queues_and_resets_recognizer(self) -> None:
        import recognizer.serial_gui as serial_gui

        class Reader:
            def __init__(self) -> None:
                self.clear_calls = 0

            def clear_pending_frames(self, *, timeout: float = 1.0) -> int:
                self.clear_calls += 1
                return 2

        class Worker:
            def __init__(self) -> None:
                self.reset_calls = 0

            def reset_recognizer(self, *, wait: bool = False, timeout: float = 1.0) -> None:
                self.reset_calls += 1

        reader = Reader()
        worker = Worker()
        result_queue: Queue[object] = Queue()
        result_queue.put(object())
        app = SimpleNamespace(
            sensor_rotation_var=FakeVar(serial_gui.SENSOR_ROTATION_OPTIONS[1]),
            sensor_rotation_state=serial_gui._ThreadSafeValue(0),
            orientation_state=serial_gui._ThreadSafeValue("原始"),
            sensor_rotation_status_var=FakeVar(),
            reader=reader,
            worker=worker,
            result_queue=result_queue,
            current_frame=np.ones((16, 16), dtype=np.float32),
            current_record=object(),
            summary_var=FakeVar(),
            _is_recording=lambda: False,
            _draw_heatmap=lambda frame: setattr(app, "drawn_frame", np.array(frame, copy=True)),
            _save_settings=lambda: setattr(app, "settings_saved", True),
        )
        app._current_sensor_rotation_degrees = lambda: int(app.sensor_rotation_state.get())
        app._update_sensor_rotation_status = lambda: serial_gui.PostureSerialApp._update_sensor_rotation_status(app)
        app._reset_direction_dependent_state = lambda: serial_gui.PostureSerialApp._reset_direction_dependent_state(app)

        serial_gui.PostureSerialApp._on_sensor_rotation_changed(app)

        self.assertEqual(app.sensor_rotation_state.get(), 180)
        self.assertEqual(reader.clear_calls, 1)
        self.assertEqual(worker.reset_calls, 1)
        self.assertEqual(result_queue.qsize(), 0)
        self.assertIsNone(app.current_frame)
        self.assertIsNone(app.current_record)
        self.assertIn("Sensor rotation: 180°", app.sensor_rotation_status_var.get())
        self.assertIn("Legacy orientation: none", app.sensor_rotation_status_var.get())
        self.assertIn("Effective transform: rotate_180", app.sensor_rotation_status_var.get())
        np.testing.assert_array_equal(app.drawn_frame, np.zeros((16, 16), dtype=np.float32))

    def test_serial_gui_settings_persist_sensor_rotation(self) -> None:
        import recognizer.serial_gui as serial_gui

        with TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            serial_gui._save_serial_gui_settings(
                settings_path,
                {
                    "sensor_rotation_degrees": 180,
                    "orientation": "原始",
                },
            )

            settings = serial_gui._load_serial_gui_settings(settings_path)

        self.assertEqual(settings["sensor_rotation_degrees"], 180)
        self.assertEqual(settings["orientation"], "原始")

    def test_serial_gui_settings_missing_or_invalid_sensor_rotation_falls_back_to_zero(self) -> None:
        import recognizer.serial_gui as serial_gui

        with TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.json"
            invalid_path = Path(tmp) / "invalid.json"
            invalid_path.write_text(
                '{"orientation": "左右翻转", "sensor_rotation_degrees": 90}',
                encoding="utf-8",
            )

            missing = serial_gui._load_serial_gui_settings(missing_path)
            invalid = serial_gui._load_serial_gui_settings(invalid_path)

        self.assertEqual(missing["sensor_rotation_degrees"], 0)
        self.assertEqual(invalid["sensor_rotation_degrees"], 0)
        self.assertEqual(invalid["orientation"], "左右翻转")

    def test_transform_diagnostics_make_legacy_double_rotation_visible(self) -> None:
        import recognizer.serial_gui as serial_gui

        none_lines = serial_gui._serial_transform_diagnostic_lines(180, "原始")
        double_lines = serial_gui._serial_transform_diagnostic_lines(180, "上下及左右翻转")

        self.assertIn("Sensor installation rotation: 180", none_lines)
        self.assertIn("Legacy data orientation: none", none_lines)
        self.assertIn("Effective transform: rotate_180", none_lines)
        self.assertIn("Worker rotation: 180", none_lines)
        self.assertIn("Recorder rotation: 180", none_lines)
        self.assertIn("Legacy data orientation: 上下及左右翻转", double_lines)
        self.assertIn("Effective transform: rotate_180 + legacy:上下及左右翻转", double_lines)

    def test_orientation_change_updates_effective_transform_status(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = SimpleNamespace(
            orientation_var=FakeVar("上下及左右翻转"),
            orientation_state=serial_gui._ThreadSafeValue("原始"),
            sensor_rotation_state=serial_gui._ThreadSafeValue(180),
            sensor_rotation_status_var=FakeVar(),
            reader=None,
            worker=None,
            result_queue=None,
            current_frame=None,
            current_record=None,
            _draw_heatmap=lambda _frame: None,
            _save_settings=lambda: None,
        )
        app._current_sensor_rotation_degrees = lambda: int(app.sensor_rotation_state.get())
        app._update_sensor_rotation_status = lambda: serial_gui.PostureSerialApp._update_sensor_rotation_status(app)
        app._reset_direction_dependent_state = lambda: serial_gui.PostureSerialApp._reset_direction_dependent_state(app)

        serial_gui.PostureSerialApp._on_orientation_changed(app)

        self.assertEqual(app.orientation_state.get(), "上下及左右翻转")
        self.assertIn("Sensor rotation: 180°", app.sensor_rotation_status_var.get())
        self.assertIn("Legacy orientation: 上下及左右翻转", app.sensor_rotation_status_var.get())
        self.assertIn("Effective transform: rotate_180 + legacy:上下及左右翻转", app.sensor_rotation_status_var.get())

    def test_connect_diagnostics_print_internal_rotation_values(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = SimpleNamespace(
            sensor_rotation_state=serial_gui._ThreadSafeValue(180),
            orientation_state=serial_gui._ThreadSafeValue("原始"),
        )
        app._current_sensor_rotation_degrees = lambda: int(app.sensor_rotation_state.get())

        with patch("builtins.print") as print_mock:
            serial_gui.PostureSerialApp._print_transform_diagnostics(app)

        printed = [call.args[0] for call in print_mock.call_args_list]
        self.assertIn("Sensor installation rotation: 180", printed)
        self.assertIn("Legacy data orientation: none", printed)
        self.assertIn("Effective transform: rotate_180", printed)
        self.assertIn("Worker rotation: 180", printed)

    def test_recognition_info_panel_expands_with_right_column(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(serial_gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Label", fake_widget_factory()),
        ):
            serial_gui.PostureSerialApp._build_status_panel(app, parent)

        result_box = next(widget for widget in FakeTkWidget.instances if widget.kwargs.get("text") == "识别信息")

        self.assertEqual(result_box.grid_kwargs["sticky"], "nsew")
        self.assertIn((1, {"weight": 1}), parent.rowconfigure_calls)

    def test_connection_info_uses_chinese_operational_labels(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(serial_gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Label", fake_widget_factory()),
        ):
            serial_gui.PostureSerialApp._build_status_panel(app, parent)

        child_texts = {widget.kwargs.get("text") for widget in FakeTkWidget.instances}

        for required_label in ("帧队列", "丢弃帧", "串口错误"):
            self.assertIn(required_label, child_texts)

    def test_visual_font_hierarchy_uses_readable_sizes(self) -> None:
        import recognizer.serial_gui as serial_gui

        sizes = serial_gui.SERIAL_FONT_SIZES

        self.assertGreaterEqual(sizes["brand_title"], 20)
        self.assertGreaterEqual(sizes["brand_subtitle"], 13)
        self.assertGreaterEqual(sizes["section_title"], 16)
        self.assertGreaterEqual(sizes["field"], 13)
        self.assertGreaterEqual(sizes["field_value"], 13)
        self.assertGreaterEqual(sizes["summary_posture"], 13)
        self.assertLessEqual(sizes["summary_posture"], 14)
        self.assertGreaterEqual(sizes["summary_meta"], 13)
        self.assertGreaterEqual(sizes["helper"], 11)
        self.assertGreaterEqual(sizes["button"], 12)
        self.assertGreaterEqual(sizes["input"], 12)
        self.assertGreaterEqual(sizes["direction"], 14)
        self.assertGreaterEqual(sizes["stats"], 12)

    def test_heatmap_side_labels_stay_close_to_actual_grid(self) -> None:
        import recognizer.serial_gui as serial_gui

        geometry = SimpleNamespace(canvas_width=620.0, canvas_height=440.0, square_size=440.0, offset_x=90.0, offset_y=0.0)
        left, right = serial_gui._heatmap_side_label_positions(geometry)

        self.assertLessEqual(abs(left[0] - geometry.offset_x), 14.0)
        self.assertLessEqual(abs(right[0] - (geometry.offset_x + geometry.square_size)), 14.0)

    def test_recognition_info_fields_are_two_column_table(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(serial_gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Label", fake_widget_factory()),
        ):
            serial_gui.PostureSerialApp._build_status_panel(app, parent)

        result_box = next(widget for widget in FakeTkWidget.instances if widget.kwargs.get("text") == "识别信息")
        expected_left_labels = [
            "当前系统状态",
            "占用状态",
            "姿势",
            "原始候选",
            "置信度",
            "第二候选",
            "置信差值",
            "识别错误",
        ]
        expected_right_labels = [
            "当前帧序号",
            "连接运行时间",
            "最近推理耗时",
            "平均推理耗时",
        ]
        long_labels = ["边界原因", "原型诊断"]
        expected_labels = expected_left_labels + expected_right_labels + long_labels

        field_widgets = [widget for widget in FakeTkWidget.instances if widget.master is result_box and widget.kwargs.get("text") in expected_labels]
        by_label = {widget.kwargs["text"]: widget.grid_kwargs for widget in field_widgets}

        self.assertEqual(set(by_label), set(expected_labels))
        for row, label in enumerate(expected_left_labels):
            self.assertEqual(by_label[label]["row"], row + 2)
            self.assertEqual(by_label[label]["column"], 0)
        for row, label in enumerate(expected_right_labels):
            self.assertEqual(by_label[label]["row"], row + 2)
            self.assertEqual(by_label[label]["column"], 3)
        for label in long_labels:
            self.assertEqual(by_label[label]["column"], 0)
            self.assertEqual(by_label[label]["columnspan"], 5)
        self.assertIn((1, {"weight": 1}), result_box.columnconfigure_calls)
        self.assertIn((4, {"weight": 1}), result_box.columnconfigure_calls)

    def test_boundary_reason_and_prototype_are_wrapped_full_width_text_blocks(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(serial_gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Label", fake_widget_factory()),
        ):
            serial_gui.PostureSerialApp._build_status_panel(app, parent)

        result_box = next(widget for widget in FakeTkWidget.instances if widget.kwargs.get("text") == "识别信息")
        long_value_widgets = [
            widget
            for widget in FakeTkWidget.instances
            if widget.master is result_box and widget.kwargs.get("textvariable") in {app.boundary_reason_var, app.prototype_var}
        ]

        self.assertEqual(len(long_value_widgets), 2)
        for widget in long_value_widgets:
            self.assertEqual(widget.grid_kwargs["column"], 0)
            self.assertEqual(widget.grid_kwargs["columnspan"], 5)
            self.assertIn("e", widget.grid_kwargs["sticky"])
            self.assertIn("w", widget.grid_kwargs["sticky"])
            self.assertGreaterEqual(int(widget.kwargs["wraplength"]), 300)

    def test_recognition_info_has_prominent_summary_area(self) -> None:
        import recognizer.serial_gui as serial_gui

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(serial_gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Frame", fake_widget_factory()),
            patch.object(serial_gui.ttk, "Label", fake_widget_factory()),
        ):
            serial_gui.PostureSerialApp._build_status_panel(app, parent)

        result_box = next(widget for widget in FakeTkWidget.instances if widget.kwargs.get("text") == "识别信息")
        child_texts = {widget.kwargs.get("text") for widget in FakeTkWidget.instances if widget.master is result_box}
        summary_values = {
            widget.kwargs.get("textvariable")
            for widget in FakeTkWidget.instances
            if widget.master is result_box
        }

        self.assertIn("当前状态", child_texts)
        self.assertIn("当前姿势", child_texts)
        self.assertIn(app.summary_posture_var, summary_values)

    def test_summary_state_labels_do_not_use_dot_markers(self) -> None:
        import recognizer.serial_gui as serial_gui

        boundary = SimpleNamespace(display_status="POSTURE", occupancy_state="HUMAN", posture=None, is_boundary=True)
        stable = SimpleNamespace(display_status="POSTURE", occupancy_state="HUMAN", posture="端正坐姿", is_boundary=False)
        empty = SimpleNamespace(display_status="EMPTY", occupancy_state="EMPTY", posture=None, is_boundary=False)

        self.assertEqual(serial_gui._summary_state_label(boundary), "边界姿势")
        self.assertEqual(serial_gui._summary_state_label(stable), "稳定识别")
        self.assertEqual(serial_gui._summary_state_label(empty), "空载")
        source = Path(serial_gui.__file__).read_text(encoding="utf-8")
        self.assertNotIn("●", source)


class SerialGuiTkLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        if os.environ.get("RUN_TK_LAYOUT_TESTS") != "1":
            self.skipTest("set RUN_TK_LAYOUT_TESTS=1 to run native Tk layout checks")

    def find_text_frame(self, root, text: str):
        stack = list(root.winfo_children())
        while stack:
            widget = stack.pop(0)
            try:
                if widget.cget("text") == text:
                    return widget
            except Exception:
                pass
            stack.extend(widget.winfo_children())
        raise AssertionError(f"could not find label frame {text!r}")

    def descendants(self, widget) -> list[object]:
        items = []
        stack = list(widget.winfo_children())
        while stack:
            child = stack.pop(0)
            items.append(child)
            stack.extend(child.winfo_children())
        return items

    def build_app(self, *, geometry: str, brand: bool = False):
        import tkinter as tk
        import recognizer.serial_gui as serial_gui

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is not available: {exc}")
        root.withdraw()
        options = {}
        if brand:
            options = {
                "brand_name": "绿联智能",
                "subtitle": "实时串口坐姿识别系统",
                "app_title": "绿联智能｜实时串口坐姿识别系统",
            }
        with patch.object(serial_gui, "list_serial_ports", return_value=[]):
            app = serial_gui.PostureSerialApp(root, **options)
        root.geometry(geometry)
        root.deiconify()
        root.update()
        return root, app

    def test_default_and_minimum_layout_do_not_request_horizontal_overflow(self) -> None:
        for geometry in ("1240x840", "1160x780"):
            with self.subTest(geometry=geometry):
                root, _app = self.build_app(geometry=geometry)
                try:
                    self.assertLessEqual(root.winfo_reqwidth(), int(geometry.split("x", maxsplit=1)[0]))
                finally:
                    root.destroy()

    def test_branded_macos_layout_does_not_request_horizontal_overflow(self) -> None:
        root, _app = self.build_app(geometry="1240x840", brand=True)
        try:
            self.assertLessEqual(root.winfo_reqwidth(), 1240)
        finally:
            root.destroy()

    def test_capture_widgets_stay_within_window_at_default_and_minimum_sizes(self) -> None:
        for geometry in ("1240x840", "1160x780"):
            with self.subTest(geometry=geometry):
                root, app = self.build_app(geometry=geometry)
                try:
                    app.capture_dir_var.set("/Users/example/a/very/long/capture/output/directory")
                    app.capture_error_var.set("一个很长的写入错误信息，不应该把控件撑出窗口")
                    root.update_idletasks()
                    capture = self.find_text_frame(root, "数据采集")
                    root_right = root.winfo_rootx() + root.winfo_width()
                    child_right = max(widget.winfo_rootx() + widget.winfo_width() for widget in self.descendants(capture))

                    self.assertLessEqual(child_right, root_right)
                finally:
                    root.destroy()

    def test_recognition_info_and_heatmap_have_usable_default_and_minimum_sizes(self) -> None:
        import tkinter as tk

        for geometry in ("1240x840", "1160x780"):
            with self.subTest(geometry=geometry):
                root, _app = self.build_app(geometry=geometry)
                try:
                    result_box = self.find_text_frame(root, "识别信息")
                    canvases = [widget for widget in self.descendants(root) if isinstance(widget, tk.Canvas)]

                    self.assertGreaterEqual(result_box.winfo_height(), result_box.winfo_reqheight())
                    self.assertGreaterEqual(canvases[0].winfo_width(), 440)
                    self.assertGreaterEqual(canvases[0].winfo_height(), 400)
                finally:
                    root.destroy()


class RecognitionWorkerTest(unittest.TestCase):
    def make_worker(
        self,
        recognizer: FakeRecognizer,
        frame_queue: Queue[np.ndarray] | None = None,
        *,
        result_queue_size: int = 3,
        clock: StepClock | None = None,
        orientation_mode: str = "原始",
        sensor_rotation_degrees: int = 0,
    ) -> tuple[RecognitionWorker, Queue[object], Queue[np.ndarray]]:
        source = frame_queue or Queue(maxsize=10)
        results: Queue[object] = Queue(maxsize=result_queue_size)
        worker = RecognitionWorker(
            frame_source=source,
            recognizer=recognizer,
            result_queue=results,
            orientation_mode=orientation_mode,
            sensor_rotation_degrees=sensor_rotation_degrees,
            connection_start_time=100.0,
            clock=clock or StepClock(),
            poll_timeout=0.02,
        )
        return worker, results, source

    def test_worker_calls_predict_in_arrival_order_when_not_backlogged(self) -> None:
        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(recognizer)

        try:
            worker.start()
            for value in (1, 2, 3):
                frames.put(np.full((16, 16), value, dtype=np.float32))
                wait_until(lambda: len(recognizer.predict_frames) >= value)
                results.get(timeout=0.5)
        finally:
            worker.stop()

        self.assertEqual([float(item[0, 0]) for item in recognizer.predict_frames], [1.0, 2.0, 3.0])

    def test_worker_applies_sensor_rotation_before_existing_orientation(self) -> None:
        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(
            recognizer,
            orientation_mode="左右翻转",
            sensor_rotation_degrees=180,
        )
        raw_physical = np.arange(256, dtype=np.float32).reshape(16, 16)

        try:
            worker.start()
            frames.put(raw_physical)
            result = results.get(timeout=0.8)
        finally:
            worker.stop()

        expected = apply_orientation(apply_sensor_rotation(raw_physical, 180), "左右翻转")
        np.testing.assert_array_equal(recognizer.predict_frames[0], expected)
        np.testing.assert_array_equal(result.frame, expected)

    def test_full_realtime_chain_uses_same_canonical_corner_frame_for_heatmap_recognizer_and_csv(self) -> None:
        import recognizer.serial_gui as serial_gui

        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(
            recognizer,
            orientation_mode="原始",
            sensor_rotation_degrees=180,
        )
        physical = asymmetric_corner_frame()
        expected = apply_sensor_and_orientation(physical, 180, "原始")

        try:
            worker.start()
            frames.put(physical)
            result = results.get(timeout=0.8)
        finally:
            worker.stop()

        variables = {
            name: FakeVar()
            for name in (
                "total_var",
                "max_var",
                "max_adc_var",
                "active_var",
                "state_var",
                "occupancy_var",
                "posture_var",
                "raw_var",
                "confidence_var",
                "second_var",
                "margin_var",
                "boundary_var",
                "boundary_reason_var",
                "prototype_var",
                "frame_index_var",
                "uptime_var",
                "inference_var",
                "average_inference_var",
                "summary_status_var",
                "summary_posture_var",
                "summary_confidence_var",
                "summary_boundary_var",
            )
        }
        app = SimpleNamespace(
            **variables,
            worker=SimpleNamespace(average_inference_ms=None),
            _draw_heatmap=lambda frame: setattr(app, "heatmap_frame", np.array(frame, copy=True)),
        )
        serial_gui.PostureSerialApp._render_result(app, result)

        with TemporaryDirectory() as tmp:
            recorder = SerialDataRecorder(queue_size=8)
            recorder.start(
                output_root=Path(tmp),
                label="端正坐姿",
                trial=1,
                serial_port="COM3",
                baudrate=460800,
                orientation="原始",
                sensor_rotation_degrees=180,
                model_version="v2_4_3_candidate",
            )
            recorder.record_parsed_frame(ParsedPressureFrame(matrix=physical, checksum=0, raw_packet=b"packet"))
            recorder.stop()
            _timestamps, csv_frames = read_sensor_csv(recorder.capture_dir / "pressure_frames.csv")

        self.assertEqual(expected[0, 0], 44.0)
        self.assertEqual(expected[0, 15], 33.0)
        self.assertEqual(expected[15, 0], 22.0)
        self.assertEqual(expected[15, 15], 11.0)
        np.testing.assert_array_equal(recognizer.predict_frames[0], expected)
        np.testing.assert_array_equal(result.frame, expected)
        np.testing.assert_array_equal(app.heatmap_frame, expected)
        np.testing.assert_array_equal(csv_frames[0], expected)

    def test_prediction_listener_receives_completed_recognition_results(self) -> None:
        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(recognizer)
        captured: list[object] = []
        worker.prediction_listener = captured.append

        try:
            worker.start()
            frames.put(np.ones((16, 16), dtype=np.float32))
            result = results.get(timeout=0.8)
            wait_until(lambda: len(captured) == 1)
        finally:
            worker.stop()

        self.assertIs(captured[0].prediction, result.prediction)
        self.assertEqual(captured[0].frame.shape, (16, 16))

    def test_worker_timestamps_increase(self) -> None:
        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(recognizer, clock=StepClock(start=100.0, step=0.25))

        try:
            worker.start()
            frames.put(np.ones((16, 16), dtype=np.float32))
            first = results.get(timeout=0.8)
            frames.put(np.full((16, 16), 2, dtype=np.float32))
            second = results.get(timeout=0.8)
        finally:
            worker.stop()

        self.assertLess(first.prediction.timestamp, second.prediction.timestamp)

    def test_result_queue_drops_oldest_when_full(self) -> None:
        recognizer = FakeRecognizer()
        worker, results, frames = self.make_worker(recognizer, result_queue_size=1)

        try:
            worker.start()
            for value in (1, 2, 3):
                frames.put(np.full((16, 16), value, dtype=np.float32))
                wait_until(lambda: len(recognizer.predict_frames) >= value)
            wait_until(lambda: worker.dropped_result_count >= 2)
        finally:
            worker.stop()

        self.assertEqual(results.qsize(), 1)
        self.assertEqual(float(results.get(timeout=0.5).frame[0, 0]), 3.0)

    def test_backlogged_frame_queue_drops_stale_unprocessed_frames(self) -> None:
        recognizer = FakeRecognizer(sleep_s=0.03)
        frames: Queue[np.ndarray] = Queue(maxsize=5)
        worker, _results, _frames = self.make_worker(recognizer, frames)

        for value in (1, 2, 3, 4, 5):
            frames.put(np.full((16, 16), value, dtype=np.float32))
        try:
            worker.start()
            wait_until(lambda: worker.dropped_input_frame_count > 0)
        finally:
            worker.stop()

        self.assertLessEqual(frames.qsize(), 5)

    def test_worker_stop_is_safe_and_idempotent(self) -> None:
        worker, _results, _frames = self.make_worker(FakeRecognizer())

        worker.start()
        worker.stop()
        worker.stop()

        self.assertFalse(worker.is_running)

    def test_predict_exception_is_saved_to_last_error(self) -> None:
        worker, _results, frames = self.make_worker(FakeRecognizer(fail_on_predict=True))

        try:
            worker.start()
            frames.put(np.ones((16, 16), dtype=np.float32))
            wait_until(lambda: worker.last_error is not None)
        finally:
            worker.stop()

        self.assertIn("predict exploded", str(worker.last_error))

    def test_reset_request_calls_recognizer_reset(self) -> None:
        recognizer = FakeRecognizer()
        worker, _results, _frames = self.make_worker(recognizer)

        try:
            worker.start()
            worker.reset_recognizer(wait=True)
        finally:
            worker.stop()

        self.assertEqual(recognizer.reset_count, 1)

    def test_calibrate_request_uses_current_frame(self) -> None:
        recognizer = FakeRecognizer()
        worker, _results, _frames = self.make_worker(recognizer)
        frame = np.arange(256, dtype=np.float32).reshape(16, 16)

        try:
            worker.start()
            worker.calibrate(frame=frame, wait=True)
        finally:
            worker.stop()

        self.assertEqual(len(recognizer.calibrate_frames), 1)
        np.testing.assert_array_equal(recognizer.calibrate_frames[0], frame)

    def test_reset_and_calibrate_do_not_run_concurrently_with_predict(self) -> None:
        recognizer = FakeRecognizer(sleep_s=0.08)
        worker, _results, frames = self.make_worker(recognizer)

        try:
            worker.start()
            frames.put(np.ones((16, 16), dtype=np.float32))
            wait_until(lambda: len(recognizer.predict_frames) == 1)
            worker.reset_recognizer(wait=True)
            worker.calibrate(frame=np.zeros((16, 16), dtype=np.float32), wait=True)
        finally:
            worker.stop()

        self.assertFalse(recognizer.concurrent_access)
        self.assertEqual(recognizer.reset_count, 1)
        self.assertEqual(len(recognizer.calibrate_frames), 1)

    def test_importing_serial_gui_does_not_open_serial_port(self) -> None:
        with patch("recognizer.frame_reader.SerialFrameReader.start", side_effect=AssertionError("opened serial")):
            module = importlib.import_module("recognizer.serial_gui")
            self.assertTrue(hasattr(module, "main"))

    def test_importing_entry_point_does_not_open_serial_port(self) -> None:
        with patch("recognizer.frame_reader.SerialFrameReader.start", side_effect=AssertionError("opened serial")):
            module = importlib.import_module("posture_serial_app_macos")
            self.assertTrue(hasattr(module, "main"))

    def test_importing_entry_points_does_not_start_capture(self) -> None:
        with patch("recognizer.serial_recorder.SerialDataRecorder.start", side_effect=AssertionError("started capture")):
            macos_module = importlib.import_module("posture_serial_app_macos")
            windows_module = importlib.import_module("posture_serial_app_windows")

        self.assertTrue(hasattr(macos_module, "main"))
        self.assertTrue(hasattr(windows_module, "main"))

    def test_default_serial_window_title_stays_unbranded(self) -> None:
        from recognizer.serial_gui import serial_window_title

        title = serial_window_title("v2_4_3_candidate")

        self.assertTrue(title.startswith("实时串口坐姿识别 - 当前模型："))
        self.assertNotIn("绿联智能", title)

    def test_macos_entry_passes_greenlink_brand_options(self) -> None:
        module = importlib.import_module("posture_serial_app_macos")

        with patch.object(module, "serial_main", return_value=0) as serial_main:
            exit_code = module.main([])

        self.assertEqual(exit_code, 0)
        serial_main.assert_called_once_with(
            [],
            brand_name="绿联智能",
            subtitle="实时串口坐姿识别系统",
            app_title="绿联智能｜实时串口坐姿识别系统",
        )

    def test_windows_entry_uses_default_unbranded_serial_gui(self) -> None:
        import recognizer.serial_gui as serial_gui

        module = importlib.import_module("posture_serial_app_windows")

        self.assertIs(module.main, serial_gui.main)


if __name__ == "__main__":
    unittest.main()
