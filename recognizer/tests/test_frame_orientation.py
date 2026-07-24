from __future__ import annotations

import unittest

import numpy as np

from recognizer.frame_orientation import SENSOR_ROTATION_0, SENSOR_ROTATION_180, apply_sensor_rotation


class SensorFrameOrientationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.arange(256, dtype=np.float32).reshape(16, 16)

    def test_rotation_zero_preserves_matrix_values_and_shape(self) -> None:
        transformed = apply_sensor_rotation(self.frame, SENSOR_ROTATION_0)

        np.testing.assert_array_equal(transformed, self.frame)
        self.assertEqual(transformed.shape, (16, 16))
        self.assertEqual(transformed.dtype, self.frame.dtype)

    def test_rotation_180_maps_all_cells_and_corners(self) -> None:
        transformed = apply_sensor_rotation(self.frame, SENSOR_ROTATION_180)

        self.assertEqual(transformed[15, 15], self.frame[0, 0])
        self.assertEqual(transformed[15, 0], self.frame[0, 15])
        self.assertEqual(transformed[0, 15], self.frame[15, 0])
        self.assertEqual(transformed[0, 0], self.frame[15, 15])
        for row in range(16):
            for col in range(16):
                self.assertEqual(transformed[row, col], self.frame[15 - row, 15 - col])

    def test_rotation_180_twice_restores_original_matrix(self) -> None:
        transformed = apply_sensor_rotation(apply_sensor_rotation(self.frame, 180), 180)

        np.testing.assert_array_equal(transformed, self.frame)

    def test_rotation_does_not_modify_input_and_returns_contiguous_copy(self) -> None:
        original = self.frame.copy()

        transformed = apply_sensor_rotation(self.frame, 180)
        transformed[0, 0] = -1

        np.testing.assert_array_equal(self.frame, original)
        self.assertTrue(transformed.flags["C_CONTIGUOUS"])

    def test_invalid_rotation_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported sensor rotation"):
            apply_sensor_rotation(self.frame, 90)


if __name__ == "__main__":
    unittest.main()
