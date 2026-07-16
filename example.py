from __future__ import annotations

import time

import numpy as np

from recognizer_api import Recognizer


def read_frame() -> np.ndarray:
    """Replace this function with the hardware reader.

    It must return one pressure matrix shaped (16, 16).
    """

    return np.zeros((16, 16), dtype=float)


def main() -> None:
    recognizer = Recognizer()

    while True:
        frame = read_frame()
        result = recognizer.predict(frame)
        print(result)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
