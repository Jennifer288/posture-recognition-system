from __future__ import annotations

import importlib
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from recognizer.frame_reader import SerialFrameReader, list_serial_ports
from recognizer.serial_gui import _recommended_port


class FakeSerial:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.closed = False
        self.read_chunks = deque([b""])

    def read(self, _size: int) -> bytes:
        return self.read_chunks.popleft() if self.read_chunks else b""

    def close(self) -> None:
        self.closed = True


class WindowsSerialCompatibilityTest(unittest.TestCase):
    def test_com_port_appears_in_scanned_ports(self) -> None:
        ports = [
            SimpleNamespace(device="COM3", description="USB Serial Device"),
            SimpleNamespace(device="COM4", description="Another USB Serial Device"),
        ]

        with patch("recognizer.frame_reader._load_serial_comports", return_value=lambda: ports):
            scanned = list_serial_ports()

        self.assertEqual([item.device for item in scanned], ["COM3", "COM4"])

    def test_com_port_can_be_recommended_and_selected(self) -> None:
        devices = ["COM3", "COM4"]

        self.assertEqual(_recommended_port(devices), "COM3")
        self.assertIn("COM4", devices)

    def test_com_port_is_preferred_over_mac_style_ports_when_present(self) -> None:
        devices = ["/dev/cu.usbserial-130", "COM5", "/dev/cu.Bluetooth-Incoming-Port"]

        self.assertEqual(_recommended_port(devices), "COM5")

    def test_macos_usbserial_recommendation_still_works(self) -> None:
        devices = ["/dev/cu.Bluetooth-Incoming-Port", "/dev/cu.usbserial-130", "debug-console"]

        self.assertEqual(_recommended_port(devices), "/dev/cu.usbserial-130")

    def test_serial_frame_reader_accepts_windows_com_port_and_uses_8n1_parameters(self) -> None:
        fake = FakeSerial()

        def factory(**kwargs):
            fake.kwargs = kwargs
            return fake

        reader = SerialFrameReader(port="COM3", serial_factory=factory, timeout=0.01)
        try:
            reader.start()
        finally:
            reader.stop()

        self.assertEqual(fake.kwargs["port"], "COM3")
        self.assertEqual(fake.kwargs["baudrate"], 460800)
        self.assertEqual(fake.kwargs["bytesize"], 8)
        self.assertEqual(fake.kwargs["parity"], "N")
        self.assertEqual(fake.kwargs["stopbits"], 1)
        self.assertFalse(fake.kwargs["xonxoff"])
        self.assertFalse(fake.kwargs["rtscts"])
        self.assertFalse(fake.kwargs["dsrdtr"])

    def test_importing_windows_live_entry_does_not_open_serial_port(self) -> None:
        with patch("recognizer.frame_reader.SerialFrameReader.start", side_effect=AssertionError("opened serial")):
            module = importlib.import_module("posture_serial_app_windows")

        self.assertTrue(hasattr(module, "main"))


if __name__ == "__main__":
    unittest.main()
