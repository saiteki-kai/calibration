import unittest

import numpy as np

from src.evaluation.metrics import binary_ece, binary_mce
from src.evaluation.visualization.utils import compute_calibration_curve


class TestCalibrationCurve(unittest.TestCase):
    def test_calibration_curve_1(self) -> None:
        """Test compute_calibration_curve using data from Table 4 - https://doi.org/10.1007/s10994-023-06336-7."""

        y_pred = np.array(
            [
                *([0.0] * 4),
                *([0.1] * 3),
                *([0.2] * 4),
                *([0.3] * 2),
                *([1 / 3] * 2),
                *([0.4] * 3),
                0.5,
                0.6,
                0.6,
                *([0.7] * 2),
                *([0.8] * 5),
                0.9,
                1.0,
            ],
            dtype=np.float64,
        )

        y_true = np.array(
            [*([0] * 9), *([1] * 2), *([0] * 4), *([1] * 3), 0, 0, 1, *([0] * 5), *([1] * 2), 1, 1],
            dtype=np.int64,
        )

        prob_true, prob_pred, bin_edges = compute_calibration_curve(y_true, y_pred, n_bins=5)

        expected_prob_true = np.array([0.18, 0.43, 0.33, 0.29, 1.00])
        expected_prob_pred = np.array([0.10, 0.35, 0.57, 0.77, 0.95])
        expected_bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

        np.testing.assert_array_almost_equal(prob_true, expected_prob_true, decimal=2)
        np.testing.assert_array_almost_equal(prob_pred, expected_prob_pred, decimal=2)
        np.testing.assert_array_almost_equal(bin_edges, expected_bin_edges, decimal=1)

        self.assertEqual(len(prob_true), 5)
        self.assertEqual(len(prob_pred), 5)
        self.assertEqual(len(bin_edges), 5)

        # Calculate ECE and MCE manually
        ece = np.average(np.abs(prob_true - prob_pred), weights=[11, 7, 3, 7, 2])
        mce = np.max(np.abs(prob_true - prob_pred))

        np.testing.assert_almost_equal(ece, 0.1873, decimal=2)
        np.testing.assert_almost_equal(binary_ece(y_pred, y_true, 5), 0.1873, decimal=3)
        np.testing.assert_almost_equal(mce, 0.48, decimal=2)
        np.testing.assert_almost_equal(binary_mce(y_pred, y_true, 5), 0.48, decimal=2)

    def test_calibration_curve_2(self) -> None:
        """Test calibration curve with a more complex distribution."""
        y_pred = np.array(
            [0.164, 0.314, 0.323, 0.411, 0.411, 0.527, 0.554, 0.619, 0.777, 0.777, 0.799, 0.891, 0.967, 0.967]
        )
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])

        # Compute calibration curve with 5 bins
        prob_true, prob_pred, bin_edges = compute_calibration_curve(y_true, y_pred, n_bins=5)

        expected_prob_true = np.array([0.000, 0.000, 0.750, 0.750, 1.000])
        expected_prob_pred = np.array([0.164, 0.319, 0.476, 0.743, 0.941])
        expected_bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

        np.testing.assert_array_almost_equal(prob_true, expected_prob_true, decimal=2)
        np.testing.assert_array_almost_equal(prob_pred, expected_prob_pred, decimal=2)
        np.testing.assert_array_almost_equal(bin_edges, expected_bin_edges, decimal=1)

        self.assertEqual(len(prob_true), 5)
        self.assertEqual(len(prob_pred), 5)
        self.assertEqual(len(bin_edges), 5)

        # Calculate and test ECE and MCE manually
        bin_counts = np.array([1, 2, 4, 4, 3])
        ece_score = np.average(np.abs(prob_true - prob_pred), weights=bin_counts)

        np.testing.assert_almost_equal(ece_score, 0.1502, decimal=3)
        np.testing.assert_almost_equal(binary_ece(y_pred, y_true, 5), 0.1502, decimal=3)

    def test_calibration_curve_3(self) -> None:
        """Test calibration curve with a single bin."""
        y_pred = np.array([0, 1, 0, 1])
        y_true = np.array([1, 1, 0, 0])

        # Compute calibration curve with 1 bin
        prob_true, prob_pred, bin_edges = compute_calibration_curve(y_true, y_pred, n_bins=1)

        expected_prob_true = np.array([0.5])
        expected_prob_pred = np.array([0.5])
        expected_bin_edges = np.array([0.0])

        np.testing.assert_array_almost_equal(prob_true, expected_prob_true, decimal=2)
        np.testing.assert_array_almost_equal(prob_pred, expected_prob_pred, decimal=2)
        np.testing.assert_array_almost_equal(bin_edges, expected_bin_edges, decimal=1)

        self.assertEqual(len(prob_true), 1)
        self.assertEqual(len(prob_pred), 1)
        self.assertEqual(len(bin_edges), 1)


if __name__ == "__main__":
    unittest.main()
