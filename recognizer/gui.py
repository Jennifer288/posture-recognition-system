from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

from .data_loader import read_sensor_csv
from .pipeline import RealtimePosturePipeline
from .prototype_bank import PrototypeBank
from .recognizer import PrototypeRecognizer


def pressure_to_color(value: float, max_value: float) -> str:
    scale = 0.0 if max_value <= 0 else min(max(value / max_value, 0.0), 1.0)
    red = int(30 + scale * 220)
    green = int(45 + (1.0 - abs(scale - 0.5) * 2.0) * 110)
    blue = int(80 + (1.0 - scale) * 150)
    return f"#{red:02x}{green:02x}{blue:02x}"


class RealtimeGuiApp:
    def __init__(
        self,
        root: object,
        pipeline: RealtimePosturePipeline,
        frames: np.ndarray,
        fps: float = 20.0,
        cell_size: int = 28,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.pipeline = pipeline
        self.frames = frames
        self.fps = float(fps)
        self.cell_size = int(cell_size)
        self.frame_index = 0
        self.running = True
        self.history: deque[str] = deque(maxlen=12)

        root.title("Recognizer V1 Demo")
        self.canvas = tk.Canvas(root, width=16 * self.cell_size, height=16 * self.cell_size, bg="#101828", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=6, padx=16, pady=16)
        self.posture_var = tk.StringVar(value="Waiting")
        self.confidence_var = tk.StringVar(value="0.00")
        self.duration_var = tk.StringVar(value="0.00s")
        self.boundary_var = tk.StringVar(value="No")
        self.phase_var = tk.StringVar(value="empty")

        self._label("Posture", 0)
        self._value(self.posture_var, 1)
        self._label("Confidence", 2)
        self._value(self.confidence_var, 3)
        self._label("Duration", 4)
        self._value(self.duration_var, 5)
        self._label("Boundary", 6)
        self._value(self.boundary_var, 7)
        self._label("Seat", 8)
        self._value(self.phase_var, 9)
        self.history_box = tk.Listbox(root, width=34, height=9)
        self.history_box.grid(row=10, column=1, padx=16, pady=(8, 16), sticky="n")
        self._draw(np.zeros((16, 16), dtype=float))
        self.root.after(int(1000 / self.fps), self._tick)

    def _label(self, text: str, row: int) -> None:
        label = self.tk.Label(self.root, text=text, anchor="w", font=("Helvetica", 12))
        label.grid(row=row, column=1, sticky="ew", padx=16)

    def _value(self, variable: object, row: int) -> None:
        label = self.tk.Label(self.root, textvariable=variable, anchor="w", font=("Helvetica", 18, "bold"))
        label.grid(row=row, column=1, sticky="ew", padx=16, pady=(0, 8))

    def _tick(self) -> None:
        if len(self.frames) == 0:
            return
        frame = self.frames[self.frame_index]
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        result = self.pipeline.update(frame)
        self._draw(frame)
        self.phase_var.set(result.seat.phase.value)
        if result.prediction is not None:
            self.posture_var.set(result.posture or "Boundary")
            self.confidence_var.set(f"{result.confidence:.2f}")
            self.duration_var.set(f"{result.duration_s:.2f}s")
            self.boundary_var.set("Yes" if result.is_boundary else "No")
            if not self.history or self.history[-1] != result.posture:
                self.history.append(result.posture or "Boundary")
                self.history_box.delete(0, self.tk.END)
                for item in reversed(self.history):
                    self.history_box.insert(self.tk.END, item)
        else:
            self.posture_var.set("Waiting")
            self.confidence_var.set("0.00")
            self.duration_var.set("0.00s")
            self.boundary_var.set("No")
        self.root.after(int(1000 / self.fps), self._tick)

    def _draw(self, frame: np.ndarray) -> None:
        self.canvas.delete("all")
        max_value = float(np.max(frame)) if frame.size else 0.0
        for row in range(16):
            for col in range(16):
                x0 = col * self.cell_size
                y0 = row * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = pressure_to_color(float(frame[row, col]), max_value)
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#172033")


def main(argv: Sequence[str] | None = None) -> int:
    import tkinter as tk

    parser = argparse.ArgumentParser(description="Simple Recognizer V1 GUI with CSV playback.")
    parser.add_argument("--model", required=True, help="Path to prototype_bank_v1.json.")
    parser.add_argument("--csv", required=True, help="Sensor CSV to replay.")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args(argv)

    bank = PrototypeBank.load(args.model)
    recognizer = PrototypeRecognizer(bank)
    _, frames = read_sensor_csv(Path(args.csv))
    root = tk.Tk()
    RealtimeGuiApp(root, RealtimePosturePipeline(recognizer, fps=args.fps), frames, fps=args.fps)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
