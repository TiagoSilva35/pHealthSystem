import unittest

import numpy as np

from src.main import extract_analog_signals, validate_channels


class TestMainUtilities(unittest.TestCase):
    def test_validate_channels_valid(self):
        self.assertEqual(validate_channels([0, 1, 5]), [0, 1, 5])
        self.assertEqual(validate_channels(["2", "3"]), [2, 3])

    def test_validate_channels_invalid_empty(self):
        with self.assertRaises(ValueError):
            validate_channels([])

    def test_validate_channels_invalid_range(self):
        with self.assertRaises(ValueError):
            validate_channels([0, 6])

    def test_extract_analog_signals(self):
        batch = np.array(
            [
                [0, 0, 0, 0, 0, 100, 200, 300],
                [1, 0, 1, 0, 1, 110, 210, 310],
            ]
        )

        analog = extract_analog_signals(batch, 3)
        expected = np.array([[100, 200, 300], [110, 210, 310]])
        np.testing.assert_array_equal(analog, expected)


if __name__ == "__main__":
    unittest.main()
