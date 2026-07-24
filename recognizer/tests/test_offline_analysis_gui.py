from __future__ import annotations

import importlib
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class FakeTkWidget:
    instances: list["FakeTkWidget"] = []

    def __init__(self, master: object | None = None, **kwargs: object) -> None:
        self.master = master
        self.kwargs = kwargs
        self.grid_kwargs: dict[str, object] | None = None
        self.rowconfigure_calls: list[tuple[int, dict[str, object]]] = []
        self.columnconfigure_calls: list[tuple[int, dict[str, object]]] = []
        self.add_calls: list[tuple[object, dict[str, object]]] = []
        self.heading_calls: list[tuple[str, dict[str, object]]] = []
        self.column_calls: list[tuple[str, dict[str, object]]] = []
        self.configure_calls: list[dict[str, object]] = []
        self.bind_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.bind_all_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.create_window_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.selected_tab: object | None = None
        self.window_id_counter = 0
        FakeTkWidget.instances.append(self)

    def grid(self, **kwargs: object) -> None:
        self.grid_kwargs = kwargs

    def rowconfigure(self, index: int, **kwargs: object) -> None:
        self.rowconfigure_calls.append((index, kwargs))

    def columnconfigure(self, index: int, **kwargs: object) -> None:
        self.columnconfigure_calls.append((index, kwargs))

    def title(self, *_args: object) -> None:
        return None

    def geometry(self, *_args: object) -> None:
        return None

    def destroy(self) -> None:
        return None

    def add(self, child: object, **kwargs: object) -> None:
        self.add_calls.append((child, kwargs))

    def heading(self, column: str, **kwargs: object) -> None:
        self.heading_calls.append((column, kwargs))

    def column(self, column: str, **kwargs: object) -> None:
        self.column_calls.append((column, kwargs))

    def configure(self, **kwargs: object) -> None:
        self.configure_calls.append(kwargs)
        self.kwargs.update(kwargs)

    def bind(self, *args: object, **kwargs: object) -> None:
        self.bind_calls.append((args, kwargs))

    def bind_all(self, *args: object, **kwargs: object) -> None:
        self.bind_all_calls.append((args, kwargs))

    def unbind_all(self, *_args: object, **_kwargs: object) -> None:
        return None

    def create_window(self, *args: object, **kwargs: object) -> int:
        self.window_id_counter += 1
        self.create_window_calls.append((args, kwargs))
        return self.window_id_counter

    def itemconfigure(self, *_args: object, **_kwargs: object) -> None:
        return None

    def bbox(self, *_args: object) -> tuple[int, int, int, int]:
        return (0, 0, 480, 640)

    def canvasy(self, value: object) -> object:
        return value

    def yview_scroll(self, *_args: object) -> None:
        return None

    def yview_moveto(self, *_args: object) -> None:
        return None

    def select(self, tab: object | None = None) -> object | None:
        if tab is not None:
            self.selected_tab = tab
        return self.selected_tab

    def selection(self) -> tuple[object, ...]:
        return ()

    def insert(self, *_args: object, **_kwargs: object) -> None:
        return None

    def get_children(self) -> tuple[object, ...]:
        return ()

    def delete(self, *_args: object) -> None:
        return None

    def yview_moveto(self, *_args: object) -> None:
        return None

    def yview(self, *_args: object) -> None:
        return None

    def xview(self, *_args: object) -> None:
        return None

    def set(self, *_args: object) -> None:
        return None


def fake_widget_factory(**extra_kwargs: object):
    def make_widget(master: object | None = None, **kwargs: object) -> FakeTkWidget:
        return FakeTkWidget(master, **{**extra_kwargs, **kwargs})

    return make_widget


class OfflineAnalysisGuiImportTest(unittest.TestCase):
    def test_importing_gui_module_does_not_create_tk_window(self) -> None:
        with patch("tkinter.Tk") as tk_mock:
            importlib.import_module("recognizer.offline_analysis_gui")

        tk_mock.assert_not_called()

    def test_importing_platform_entries_does_not_create_tk_window(self) -> None:
        with patch("tkinter.Tk") as tk_mock:
            importlib.import_module("posture_offline_serial_app_macos")
            importlib.import_module("posture_offline_serial_app_windows")

        tk_mock.assert_not_called()

    def test_macos_entry_passes_brand_title_and_v243_model(self) -> None:
        import posture_offline_serial_app_macos as mac_entry

        with patch.object(mac_entry, "offline_main", return_value=0) as offline_main:
            rc = mac_entry.main([])

        self.assertEqual(rc, 0)
        offline_main.assert_called_once()
        kwargs = offline_main.call_args.kwargs
        self.assertEqual(kwargs["brand_name"], "绿联智能")
        self.assertEqual(kwargs["subtitle"], "离线串口坐姿分析软件")
        self.assertEqual(kwargs["app_title"], "绿联智能｜离线串口坐姿分析软件")
        self.assertEqual(kwargs["model_version"], "v2_4_3_candidate")

    def test_windows_entry_uses_default_unbranded_gui(self) -> None:
        import posture_offline_serial_app_windows as windows_entry

        with patch.object(windows_entry, "offline_main", return_value=0) as offline_main:
            rc = windows_entry.main([])

        self.assertEqual(rc, 0)
        offline_main.assert_called_once_with([], model_version="v2_4_3_candidate")


class OfflineAnalysisGuiStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTkWidget.instances.clear()

    def test_gui_constants_match_product_scope(self) -> None:
        import recognizer.offline_analysis_gui as gui

        self.assertEqual(gui.APP_NAME, "离线串口坐姿分析软件")
        self.assertEqual(gui.DEFAULT_WINDOW_GEOMETRY, "1240x840")
        self.assertEqual(gui.MIN_WINDOW_SIZE, (1100, 700))
        self.assertGreaterEqual(gui.OFFLINE_FONT_SIZES["field"], 13)
        self.assertGreaterEqual(gui.OFFLINE_FONT_SIZES["section_title"], 16)

    def test_initial_window_geometry_respects_screen_safe_height(self) -> None:
        import recognizer.offline_analysis_gui as gui

        small = gui.calculate_initial_window_geometry(1440, 780)
        self.assertEqual((small.width, small.height), (1240, 700))
        self.assertLess(small.height, 840)
        self.assertGreaterEqual(small.x, 0)
        self.assertGreaterEqual(small.y, 0)

        roomy = gui.calculate_initial_window_geometry(1440, 1000)
        self.assertEqual((roomy.width, roomy.height), (1240, 840))

    def test_configure_root_allows_vertical_resize_and_uses_dynamic_geometry(self) -> None:
        import recognizer.offline_analysis_gui as gui

        root = Mock()
        root.winfo_screenwidth.return_value = 1440
        root.winfo_screenheight.return_value = 780
        app = SimpleNamespace(root=root, app_title="离线串口坐姿分析软件")

        gui.PostureOfflineSerialApp._configure_root(app)

        root.resizable.assert_called_once_with(True, True)
        root.minsize.assert_called_once_with(1100, 700)
        geometry_arg = root.geometry.call_args.args[0]
        self.assertTrue(geometry_arg.startswith("1240x700+"))
        root.rowconfigure.assert_any_call(2, weight=1)

    def test_playback_delay_supports_required_speeds(self) -> None:
        import recognizer.offline_analysis_gui as gui

        self.assertGreater(gui._playback_delay_ms("0.5×", 20.0), gui._playback_delay_ms("1×", 20.0))
        self.assertGreater(gui._playback_delay_ms("1×", 20.0), gui._playback_delay_ms("2×", 20.0))
        self.assertEqual(gui._playback_delay_ms("最大速度", 20.0), 1)

    def test_main_creates_app_only_when_called(self) -> None:
        import recognizer.offline_analysis_gui as gui

        fake_root = Mock()
        with (
            patch.object(gui.tk, "Tk", return_value=fake_root) as tk_mock,
            patch.object(gui, "PostureOfflineSerialApp") as app_mock,
        ):
            rc = gui.main([])

        self.assertEqual(rc, 0)
        tk_mock.assert_called_once()
        app_mock.assert_called_once()
        self.assertEqual(app_mock.call_args.kwargs["model_version"], "v2_4_3_candidate")
        fake_root.mainloop.assert_called_once()

    def make_fake_app(self) -> SimpleNamespace:
        import recognizer.offline_analysis_gui as gui

        root = SimpleNamespace(after_idle=Mock())
        app = SimpleNamespace(
            root=root,
            path_var=object(),
            input_file_var=object(),
            input_directory_var=object(),
            input_type_var=object(),
            file_size_var=object(),
            metadata_path_var=object(),
            manual_label_var=object(),
            manual_label_trial_var=object(),
            trial_var=object(),
            capture_time_var=object(),
            orientation_var=object(),
            sensor_rotation_var=object(),
            fps_var=object(),
            fps_source_var=object(),
            total_bytes_var=object(),
            valid_packets_var=object(),
            invalid_packets_var=object(),
            discarded_bytes_var=object(),
            invalid_lines_var=object(),
            data_complete_var=object(),
            calibration_var=object(),
            overall_var=object(),
            status_var=object(),
            dominant_var=object(),
            share_var=object(),
            mean_conf_var=object(),
            boundary_rate_var=object(),
            human_duration_var=object(),
            stable_duration_var=object(),
            label_match_var=object(),
            warnings_var=object(),
            expand_statistics_window=lambda: None,
            _on_segment_selected=lambda _event: None,
            _update_file_info_wraplength=lambda _event=None: None,
            _bind_treeview_mousewheel=lambda _tree: None,
        )
        app._field = lambda *args, **kwargs: gui.PostureOfflineSerialApp._field(app, *args, **kwargs)
        app._build_tree_table = lambda *args, **kwargs: gui.PostureOfflineSerialApp._build_tree_table(app, *args, **kwargs)
        app._build_scrollable_right_page = lambda *args, **kwargs: gui.PostureOfflineSerialApp._build_scrollable_right_page(app, *args, **kwargs)
        app._build_file_info = lambda *args, **kwargs: gui.PostureOfflineSerialApp._build_file_info(app, *args, **kwargs)
        app._build_summary = lambda *args, **kwargs: gui.PostureOfflineSerialApp._build_summary(app, *args, **kwargs)
        app._build_tables = lambda *args, **kwargs: gui.PostureOfflineSerialApp._build_tables(app, *args, **kwargs)
        return app

    def make_fake_left_panel_app(self) -> SimpleNamespace:
        app = SimpleNamespace(
            total_pressure_var=object(),
            max_pressure_var=object(),
            active_points_var=object(),
            frame_info_var=object(),
            frame_state_var=object(),
            frame_posture_var=object(),
            frame_confidence_var=object(),
            frame_boundary_var=object(),
            jump_start=lambda: None,
            previous_frame=lambda: None,
            play=lambda: None,
            pause=lambda: None,
            next_frame=lambda: None,
            jump_end=lambda: None,
            _on_frame_scale=lambda _value: None,
            _render_current_frame=lambda: None,
        )
        return app

    def test_right_panel_uses_single_scrollable_page_without_paned_window(self) -> None:
        import recognizer.offline_analysis_gui as gui

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(gui.ttk, "Frame", fake_widget_factory()),
            patch.object(gui.ttk, "PanedWindow", Mock(side_effect=AssertionError("right panel must not use PanedWindow"))),
            patch.object(gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(gui.ttk, "Label", fake_widget_factory()),
            patch.object(gui.ttk, "Button", fake_widget_factory()),
            patch.object(gui.ttk, "Notebook", fake_widget_factory()),
            patch.object(gui.ttk, "Treeview", fake_widget_factory()),
            patch.object(gui.ttk, "Scrollbar", fake_widget_factory()),
            patch.object(gui.tk, "Canvas", fake_widget_factory()),
        ):
            gui.PostureOfflineSerialApp._build_right_panel(app, parent)

        self.assertIn((0, {"weight": 1}), parent.rowconfigure_calls)
        self.assertIsNotNone(getattr(app, "right_scroll_canvas"))
        self.assertEqual(app.right_scroll_canvas.grid_kwargs["sticky"], "nsew")
        self.assertEqual(app.right_scrollbar.grid_kwargs["sticky"], "ns")
        self.assertTrue(any("yscrollcommand" in call for call in app.right_scroll_canvas.configure_calls))
        self.assertTrue(app.right_scroll_canvas.create_window_calls)
        self.assertIs(app.right_scroll_content.master, app.right_scroll_canvas)
        canvas_bindings = {args[0] for args, _kwargs in app.right_scroll_canvas.bind_calls}
        self.assertIn("<Configure>", canvas_bindings)
        self.assertNotIn("<Enter>", canvas_bindings)
        self.assertNotIn("<Leave>", canvas_bindings)
        self.assertIs(app.tables_container, app.right_scroll_content)

    def test_tables_are_stretchy_and_have_scrollbars_with_minimum_visible_rows(self) -> None:
        import recognizer.offline_analysis_gui as gui

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(gui.ttk, "Frame", fake_widget_factory()),
            patch.object(gui.ttk, "Label", fake_widget_factory()),
            patch.object(gui.ttk, "Button", fake_widget_factory()),
            patch.object(gui.ttk, "Notebook", fake_widget_factory()),
            patch.object(gui.ttk, "Treeview", fake_widget_factory()),
            patch.object(gui.ttk, "Scrollbar", fake_widget_factory()),
        ):
            gui.PostureOfflineSerialApp._build_tables(app, parent)

        self.assertIs(app.tables_container, parent)
        self.assertTrue(any(index == 3 and options.get("weight") == 0 for index, options in parent.rowconfigure_calls))
        self.assertIn((0, {"weight": 1}), parent.columnconfigure_calls)
        self.assertEqual(app.notebook.grid_kwargs["sticky"], "nsew")
        self.assertGreaterEqual(int(parent.rowconfigure_calls[-1][1].get("minsize", 0)), gui.RIGHT_NOTEBOOK_MIN_HEIGHT)
        self.assertEqual(app.stats_table.grid_kwargs["sticky"], "nsew")
        self.assertEqual(app.segment_table.grid_kwargs["sticky"], "nsew")
        self.assertGreaterEqual(int(app.stats_table.kwargs["height"]), 8)
        self.assertGreaterEqual(int(app.segment_table.kwargs["height"]), 8)
        self.assertIsNotNone(app.stats_v_scrollbar)
        self.assertIsNotNone(app.stats_h_scrollbar)
        self.assertIsNotNone(app.segment_v_scrollbar)
        self.assertIsNotNone(app.segment_h_scrollbar)
        self.assertTrue(any("yscrollcommand" in call for call in app.stats_table.configure_calls))
        self.assertTrue(any("xscrollcommand" in call for call in app.segment_table.configure_calls))
        self.assertEqual(app.expand_stats_button.kwargs.get("text"), "展开统计")
        stats_width = sum(int(call[1]["width"]) for call in app.stats_table.column_calls)
        segment_width = sum(int(call[1]["width"]) for call in app.segment_table.column_calls)
        self.assertLessEqual(stats_width, 430)
        self.assertLessEqual(segment_width, 470)

    def test_playback_controls_wrap_to_multiple_rows_to_avoid_horizontal_overflow(self) -> None:
        import recognizer.offline_analysis_gui as gui

        app = self.make_fake_left_panel_app()
        parent = FakeTkWidget()
        with (
            patch.object(gui.ttk, "Frame", fake_widget_factory()),
            patch.object(gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(gui.ttk, "Label", fake_widget_factory()),
            patch.object(gui.ttk, "Button", fake_widget_factory()),
            patch.object(gui.ttk, "Scale", fake_widget_factory()),
            patch.object(gui.ttk, "Combobox", fake_widget_factory()),
            patch.object(gui.tk, "Canvas", fake_widget_factory()),
            patch.object(gui.tk, "StringVar", lambda value=None: SimpleNamespace(get=lambda: value, set=lambda _value: None)),
        ):
            gui.PostureOfflineSerialApp._build_left_panel(app, parent)

        button_positions = {
            widget.kwargs.get("text"): widget.grid_kwargs
            for widget in FakeTkWidget.instances
            if widget.kwargs.get("text") in {"开头", "上一帧", "播放", "暂停", "下一帧", "结尾"}
        }
        self.assertEqual(button_positions["开头"]["row"], 0)
        self.assertEqual(button_positions["播放"]["row"], 0)
        self.assertEqual(button_positions["暂停"]["row"], 1)
        self.assertEqual(button_positions["下一帧"]["row"], 1)
        scales = [widget for widget in FakeTkWidget.instances if widget.kwargs.get("orient") == "horizontal"]
        self.assertEqual(scales[0].grid_kwargs["row"], 0)
        self.assertEqual(scales[0].grid_kwargs["column"], 3)

    def test_mousewheel_units_support_small_macos_delta_windows_and_linux(self) -> None:
        import recognizer.offline_analysis_gui as gui

        self.assertEqual(gui._mousewheel_scroll_units(SimpleNamespace(delta=1)), -1)
        self.assertEqual(gui._mousewheel_scroll_units(SimpleNamespace(delta=-1)), 1)
        self.assertEqual(gui._mousewheel_scroll_units(SimpleNamespace(delta=240)), -2)
        self.assertEqual(gui._mousewheel_scroll_units(SimpleNamespace(delta=-240)), 2)
        self.assertEqual(gui._mousewheel_scroll_units(SimpleNamespace(delta=0, num=4)), -1)
        self.assertEqual(gui._mousewheel_scroll_units(SimpleNamespace(delta=0, num=5)), 1)

    def test_top_information_fields_are_preserved_in_compact_layouts(self) -> None:
        import recognizer.offline_analysis_gui as gui

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(gui.ttk, "Label", fake_widget_factory()),
            patch.object(gui.ttk, "Button", fake_widget_factory()),
        ):
            gui.PostureOfflineSerialApp._build_file_info(app, parent)
            gui.PostureOfflineSerialApp._build_summary(app, parent)

        labels = {widget.kwargs.get("text") for widget in FakeTkWidget.instances}
        file_info_box = next(widget for widget in FakeTkWidget.instances if widget.kwargs.get("text") == "文件与协议信息")
        file_info_labels = {widget.kwargs.get("text") for widget in FakeTkWidget.instances if widget.master is file_info_box}
        for label in {
            "输入文件",
            "所在目录",
            "输入类型",
            "人工标签与采集次数",
            "采集时间",
            "数据方向",
            "FPS",
            "有效协议包",
            "无效协议包",
            "数据完整状态",
            "校准状态",
        }:
            self.assertIn(label, file_info_labels)
        for label in {
            "综合判定",
            "结果状态",
            "主姿势",
            "主姿势占比",
            "平均置信度",
            "边界率",
            "人体时长",
            "稳定识别时长",
            "是否一致",
            "质量警告",
        }:
            self.assertIn(label, labels)
        for removed_label in {
            "输入路径",
            "metadata路径",
            "文件大小",
            "人工标签",
            "采集次数",
            "方向",
            "FPS来源",
            "原始字节数",
            "丢弃字节",
            "无效TXT行",
        }:
            self.assertNotIn(removed_label, file_info_labels)
        self.assertNotIn("查看完整警告", labels)

        full_width_vars = {
            app.input_file_var,
            app.input_directory_var,
            app.calibration_var,
        }
        full_width_labels = [
            widget
            for widget in FakeTkWidget.instances
            if widget.kwargs.get("textvariable") in full_width_vars
        ]
        self.assertEqual(len(full_width_labels), 3)
        for value_label in full_width_labels:
            self.assertEqual(value_label.grid_kwargs["column"], 1)
            self.assertGreaterEqual(int(value_label.grid_kwargs["columnspan"]), 3)
            self.assertEqual(value_label.grid_kwargs["sticky"], "ew")
            self.assertEqual(value_label.kwargs.get("anchor"), "w")

        warning_labels = [
            widget
            for widget in FakeTkWidget.instances
            if widget.kwargs.get("textvariable") is app.warnings_var
        ]
        self.assertEqual(len(warning_labels), 1)
        warning_label = warning_labels[0]
        self.assertGreaterEqual(int(warning_label.grid_kwargs["columnspan"]), 5)
        self.assertEqual(warning_label.kwargs.get("justify"), "left")
        self.assertEqual(warning_label.kwargs.get("anchor"), "nw")
        self.assertGreater(int(warning_label.kwargs.get("wraplength", 0)), 0)

    def test_file_info_wraplength_updates_for_path_and_calibration_labels(self) -> None:
        import recognizer.offline_analysis_gui as gui

        labels = [Mock(), Mock(), Mock()]
        app = SimpleNamespace(file_info_wrap_labels=labels)

        gui.PostureOfflineSerialApp._update_file_info_wraplength(app, SimpleNamespace(width=720))

        for label in labels:
            wraplength = label.configure.call_args.kwargs["wraplength"]
            self.assertGreaterEqual(wraplength, 320)
            self.assertLessEqual(wraplength, 590)

    def test_display_formatters_keep_file_info_readable(self) -> None:
        import recognizer.offline_analysis_gui as gui

        self.assertEqual(gui.format_capture_time("2026-07-22T13:41:07.116371"), "2026-07-22 13:41:07")
        self.assertEqual(gui.format_capture_time("2026/07/22_13:41:07.116371"), "2026-07-22 13:41:07")
        self.assertEqual(gui.format_manual_label("test", 1), "test · 第1次")
        self.assertEqual(gui.format_manual_label("test", None), "test")
        self.assertEqual(gui.format_manual_label(None, None), "—")
        self.assertEqual(gui.format_direction(None), "未记录")
        self.assertEqual(gui.format_integrity_status(True), "完整")
        self.assertEqual(gui.format_integrity_status(False), "不完整")
        self.assertEqual(gui.format_integrity_status(None), "—")
        self.assertEqual(gui.format_calibration_status("NO_RELIABLE_EMPTY_BASELINE"), "未找到可靠空载基线")
        self.assertEqual(gui.format_calibration_status("CALIBRATED_FROM_INITIAL_EMPTY"), "已使用开头空载数据校准")
        self.assertEqual(gui.format_calibration_status("CALIBRATION_SKIPPED"), "未执行空载校准")
        self.assertEqual(gui.format_calibration_status("CALIBRATION_FAILED"), "空载校准失败")

    def test_update_file_info_uses_compact_values_and_chinese_statuses(self) -> None:
        import recognizer.offline_analysis_gui as gui

        variables = {
            name: Mock()
            for name in (
                "path_var",
                "input_file_var",
                "input_directory_var",
                "input_type_var",
                "file_size_var",
                "fps_var",
                "fps_source_var",
                "orientation_var",
                "sensor_rotation_var",
                "total_bytes_var",
                "valid_packets_var",
                "invalid_packets_var",
                "discarded_bytes_var",
                "invalid_lines_var",
                "data_complete_var",
                "calibration_var",
                "manual_label_var",
                "manual_label_trial_var",
                "trial_var",
                "capture_time_var",
            )
        }
        app = SimpleNamespace(**variables)
        input_path = Path("tmp") / "capture" / "raw_stream.bin"
        result = SimpleNamespace(
            input_path=str(input_path),
            input_type="BIN",
            fps=20.05,
            fps_source="metadata",
            sensor_rotation_degrees=180,
            orientation="原始",
            metadata={"label": "test", "trial": 1, "start_time": "2026-07-22T13:41:07.116371", "capture_completed": True},
            parser_stats={"total_bytes": 999, "valid_packets": 252, "invalid_packets": 0, "discarded_bytes": 0, "invalid_text_line_count": 0},
            invalid_text_lines=[],
            calibration_info=SimpleNamespace(calibration_status="NO_RELIABLE_EMPTY_BASELINE"),
        )

        gui.PostureOfflineSerialApp._update_file_and_parse_info(app, result)

        app.input_file_var.set.assert_called_once_with(input_path.name)
        app.input_directory_var.set.assert_called_once_with(str(input_path.parent))
        app.sensor_rotation_var.set.assert_called_once()
        app.manual_label_trial_var.set.assert_called_once_with("test · 第1次")
        app.capture_time_var.set.assert_called_once_with("2026-07-22 13:41:07")
        app.data_complete_var.set.assert_called_once_with("完整")
        app.calibration_var.set.assert_called_once_with("未找到可靠空载基线")

    def test_quality_warning_display_adds_abnormal_parser_stats(self) -> None:
        import recognizer.offline_analysis_gui as gui

        result = SimpleNamespace(
            summary=SimpleNamespace(warnings=["已有警告"]),
            parser_stats={
                "invalid_packets": 2,
                "discarded_bytes": 125,
                "invalid_text_line_count": 1,
                "trailing_incomplete_bytes": 3,
            },
        )

        warnings = gui._warnings_for_result_display(result)

        self.assertIn("已有警告", warnings)
        self.assertIn("存在2个无效协议包", warnings)
        self.assertIn("丢弃了125个非协议字节", warnings)
        self.assertIn("串口文本中存在1个无效行", warnings)
        self.assertIn("文件末尾存在未完成的数据包", warnings)

    def test_quality_warning_popup_button_and_handler_are_removed(self) -> None:
        import recognizer.offline_analysis_gui as gui

        self.assertFalse(hasattr(gui.PostureOfflineSerialApp, "show_full_warnings"))

        app = self.make_fake_app()
        parent = FakeTkWidget()
        with (
            patch.object(gui.ttk, "LabelFrame", fake_widget_factory()),
            patch.object(gui.ttk, "Label", fake_widget_factory()),
            patch.object(gui.ttk, "Button", fake_widget_factory()),
        ):
            gui.PostureOfflineSerialApp._build_summary(app, parent)

        button_texts = {
            widget.kwargs.get("text")
            for widget in FakeTkWidget.instances
            if "command" in widget.kwargs
        }
        self.assertNotIn("查看完整警告", button_texts)
        self.assertFalse(hasattr(app, "full_warnings_button"))

    def test_warning_display_formats_all_warnings_without_truncation(self) -> None:
        import recognizer.offline_analysis_gui as gui

        warnings = [
            "pressure_frames.csv存在，但离线分析不会将其作为算法输入",
            "recognition_results.csv存在，但离线分析不会将其作为算法答案",
            "文件末尾存在未完成的串口半包，并且这条很长的警告必须完整显示不能被省略号截断",
        ]

        display = gui._format_warnings_for_display(warnings)

        for warning in warnings:
            self.assertIn(warning, display)
        self.assertIn("\n", display)
        self.assertNotIn("...", display)
        self.assertEqual(gui._format_warnings_for_display([]), "无")

    def test_update_summary_sets_full_multiline_warning_text(self) -> None:
        import recognizer.offline_analysis_gui as gui

        long_warning = "A" * 220
        app = SimpleNamespace(
            overall_var=Mock(),
            status_var=Mock(),
            dominant_var=Mock(),
            share_var=Mock(),
            mean_conf_var=Mock(),
            boundary_rate_var=Mock(),
            human_duration_var=Mock(),
            stable_duration_var=Mock(),
            label_match_var=Mock(),
            warnings_var=Mock(),
        )
        result = SimpleNamespace(
            summary=SimpleNamespace(
                overall_posture="端正坐姿",
                result_status="SUCCESS",
                dominant_posture="端正坐姿",
                dominant_posture_share=0.9,
                mean_confidence=0.8,
                boundary_rate=0.1,
                human_duration_s=10.0,
                stable_posture_duration_s=8.0,
                label_matches_overall=True,
                warnings=["第一条警告", long_warning],
            )
        )

        gui.PostureOfflineSerialApp._update_summary(app, result)

        warning_text = app.warnings_var.set.call_args.args[0]
        self.assertIn("第一条警告", warning_text)
        self.assertIn(long_warning, warning_text)
        self.assertIn("\n", warning_text)
        self.assertNotIn("...", warning_text)

    def test_quality_warning_wraplength_updates_with_summary_width(self) -> None:
        import recognizer.offline_analysis_gui as gui

        label = Mock()
        app = SimpleNamespace(quality_warning_label=label)

        gui.PostureOfflineSerialApp._update_quality_warning_wraplength(app, SimpleNamespace(width=720))

        wraplength = label.configure.call_args.kwargs["wraplength"]
        self.assertGreaterEqual(wraplength, 320)
        self.assertLessEqual(wraplength, 620)

    def test_analysis_completion_selects_statistics_tab_and_restores_table_pane(self) -> None:
        import recognizer.offline_analysis_gui as gui

        app = SimpleNamespace(
            current_frame_index=99,
            frame_scale=Mock(),
            progress_var=Mock(),
            stats_table=Mock(),
            notebook=Mock(),
            stats_tab=object(),
            _update_file_and_parse_info=Mock(),
            _update_summary=Mock(),
            _update_tables=Mock(),
            _render_current_frame=Mock(),
            _set_idle_controls=Mock(),
            _focus_statistics_panel=Mock(),
        )
        result = SimpleNamespace(
            frames=[object()] * 10,
            frame_predictions=[object()] * 10,
            summary=SimpleNamespace(overall_posture="端正坐姿"),
        )

        gui.PostureOfflineSerialApp._handle_analysis_done(app, result)

        app._focus_statistics_panel.assert_called_once()

    def test_focus_statistics_panel_selects_stats_tab_without_forcing_outer_scroll(self) -> None:
        import recognizer.offline_analysis_gui as gui

        app = SimpleNamespace(
            notebook=Mock(),
            stats_tab=object(),
            stats_table=Mock(),
            right_scroll_canvas=Mock(),
        )

        gui.PostureOfflineSerialApp._focus_statistics_panel(app)

        app.notebook.select.assert_called_once_with(app.stats_tab)
        app.stats_table.yview_moveto.assert_called_once_with(0)
        app.right_scroll_canvas.yview_moveto.assert_not_called()

    def test_global_mousewheel_binding_uses_coordinate_routing_not_enter_leave_state(self) -> None:
        import recognizer.offline_analysis_gui as gui

        root = Mock()
        app = SimpleNamespace(root=root, _global_mousewheel_bound=False)

        gui.PostureOfflineSerialApp._bind_global_mousewheel(app)

        bindings = [call.args[0] for call in root.bind_all.call_args_list]
        self.assertIn("<MouseWheel>", bindings)
        self.assertIn("<Button-4>", bindings)
        self.assertIn("<Button-5>", bindings)
        self.assertFalse(hasattr(gui.PostureOfflineSerialApp, "_activate_right_scroll"))
        self.assertFalse(hasattr(gui.PostureOfflineSerialApp, "_deactivate_right_scroll"))
        self.assertFalse(hasattr(gui.PostureOfflineSerialApp, "_pointer_inside_right_scroll_region"))

    def test_mousewheel_routes_right_label_and_small_macos_delta_to_canvas(self) -> None:
        import recognizer.offline_analysis_gui as gui

        right = SimpleNamespace(master=None)
        label = SimpleNamespace(master=right)
        canvas = Mock()
        root = Mock(winfo_containing=Mock(return_value=label))
        app = SimpleNamespace(root=root, right_scroll_container=right, right_scroll_canvas=canvas)

        result = gui.PostureOfflineSerialApp._route_mousewheel(app, SimpleNamespace(x_root=10, y_root=20, delta=-1))

        canvas.yview_scroll.assert_called_once_with(1, "units")
        self.assertEqual(result, "break")

    def test_mousewheel_routes_linux_button_events_to_canvas(self) -> None:
        import recognizer.offline_analysis_gui as gui

        right = SimpleNamespace(master=None)
        label = SimpleNamespace(master=right)
        canvas = Mock()
        root = Mock(winfo_containing=Mock(return_value=label))
        app = SimpleNamespace(root=root, right_scroll_container=right, right_scroll_canvas=canvas)

        gui.PostureOfflineSerialApp._route_mousewheel(app, SimpleNamespace(x_root=10, y_root=20, delta=0, num=4))
        canvas.yview_scroll.assert_called_once_with(-1, "units")

    def test_mousewheel_does_not_scroll_right_canvas_from_left_panel(self) -> None:
        import recognizer.offline_analysis_gui as gui

        right = SimpleNamespace(master=None)
        left = SimpleNamespace(master=None)
        canvas = Mock()
        root = Mock(winfo_containing=Mock(return_value=left))
        app = SimpleNamespace(root=root, right_scroll_container=right, right_scroll_canvas=canvas)

        result = gui.PostureOfflineSerialApp._route_mousewheel(app, SimpleNamespace(x_root=10, y_root=20, delta=-120))

        canvas.yview_scroll.assert_not_called()
        self.assertIsNone(result)

    def test_mousewheel_scrolls_treeview_only_when_tree_can_scroll(self) -> None:
        import recognizer.offline_analysis_gui as gui

        right = SimpleNamespace(master=None)
        tree = Mock()
        tree.master = right
        tree.yview.return_value = (0.2, 0.8)
        canvas = Mock()
        app = SimpleNamespace(right_scroll_container=right, right_scroll_canvas=canvas)

        result = gui.PostureOfflineSerialApp._scroll_treeview_for_mousewheel(app, tree, SimpleNamespace(delta=-120))

        tree.yview_scroll.assert_called_once_with(1, "units")
        canvas.yview_scroll.assert_not_called()
        self.assertEqual(result, "break")

    def test_mousewheel_scrolls_outer_canvas_when_tree_cannot_scroll(self) -> None:
        import recognizer.offline_analysis_gui as gui

        right = SimpleNamespace(master=None)
        tree = Mock()
        tree.master = right
        tree.yview.return_value = (0.0, 1.0)
        canvas = Mock()
        app = SimpleNamespace(right_scroll_container=right, right_scroll_canvas=canvas)

        result = gui.PostureOfflineSerialApp._scroll_treeview_for_mousewheel(app, tree, SimpleNamespace(delta=-120))

        tree.yview_scroll.assert_not_called()
        canvas.yview_scroll.assert_called_once_with(1, "units")
        self.assertEqual(result, "break")

    def test_mousewheel_scrolls_outer_canvas_when_tree_is_at_edges(self) -> None:
        import recognizer.offline_analysis_gui as gui

        right = SimpleNamespace(master=None)
        canvas = Mock()
        app = SimpleNamespace(right_scroll_container=right, right_scroll_canvas=canvas)

        top_tree = Mock()
        top_tree.master = right
        top_tree.yview.return_value = (0.0, 0.5)
        gui.PostureOfflineSerialApp._scroll_treeview_for_mousewheel(app, top_tree, SimpleNamespace(delta=120))
        canvas.yview_scroll.assert_called_once_with(-1, "units")
        top_tree.yview_scroll.assert_not_called()

        canvas.reset_mock()
        bottom_tree = Mock()
        bottom_tree.master = right
        bottom_tree.yview.return_value = (0.5, 1.0)
        gui.PostureOfflineSerialApp._scroll_treeview_for_mousewheel(app, bottom_tree, SimpleNamespace(delta=-120))
        canvas.yview_scroll.assert_called_once_with(1, "units")
        bottom_tree.yview_scroll.assert_not_called()

    def test_window_close_unbinds_global_mousewheel(self) -> None:
        import recognizer.offline_analysis_gui as gui

        root = Mock()
        app = SimpleNamespace(
            cancel_event=Mock(),
            pause=Mock(),
            after_id=None,
            root=root,
            _global_mousewheel_bound=True,
        )

        gui.PostureOfflineSerialApp._on_close(app)

        unbound = [call.args[0] for call in root.unbind_all.call_args_list]
        self.assertIn("<MouseWheel>", unbound)
        self.assertIn("<Button-4>", unbound)
        self.assertIn("<Button-5>", unbound)
        root.destroy.assert_called_once()

    def test_window_diagnostics_report_resize_and_pane_state(self) -> None:
        import recognizer.offline_analysis_gui as gui

        root = Mock()
        root.geometry.return_value = "1240x700+100+40"
        root.winfo_width.return_value = 1240
        root.winfo_height.return_value = 700
        root.winfo_screenwidth.return_value = 1440
        root.winfo_screenheight.return_value = 780
        root.minsize.return_value = (1100, 700)
        root.resizable.return_value = (True, True)
        app = SimpleNamespace(
            root=root,
            right_scroll_canvas=Mock(winfo_height=Mock(return_value=480)),
            right_scroll_content=Mock(winfo_reqheight=Mock(return_value=760)),
        )

        diagnostics = gui.PostureOfflineSerialApp.window_layout_diagnostics(app)

        self.assertEqual(diagnostics["actual_geometry"], "1240x700+100+40")
        self.assertEqual(diagnostics["minimum_height"], 700)
        self.assertTrue(diagnostics["resizable_height"])
        self.assertEqual(diagnostics["right_scroll_canvas_height"], 480)
        self.assertEqual(diagnostics["right_scroll_content_height"], 760)

    def test_expand_statistics_window_can_create_and_close_without_reusing_main_treeviews(self) -> None:
        import recognizer.offline_analysis_gui as gui

        app = SimpleNamespace(root=object(), analysis_result=None)
        app._build_tree_table = lambda *args, **kwargs: gui.PostureOfflineSerialApp._build_tree_table(app, *args, **kwargs)
        app._bind_treeview_mousewheel = lambda _tree: None
        with (
            patch.object(gui.tk, "Toplevel", fake_widget_factory()),
            patch.object(gui.ttk, "Frame", fake_widget_factory()),
            patch.object(gui.ttk, "Notebook", fake_widget_factory()),
            patch.object(gui.ttk, "Treeview", fake_widget_factory()),
            patch.object(gui.ttk, "Scrollbar", fake_widget_factory()),
        ):
            gui.PostureOfflineSerialApp.expand_statistics_window(app)

        toplevel = FakeTkWidget.instances[0]
        self.assertEqual(toplevel.kwargs, {})


if __name__ == "__main__":
    unittest.main()
