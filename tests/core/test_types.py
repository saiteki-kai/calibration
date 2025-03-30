import tempfile
import unittest

from pathlib import Path

import numpy as np

from src.core.types import CalibratorOutput, ClassifierOutput


class TestClassifierOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_classifier_output_save_load(self) -> None:
        label_probs = np.array([0.1, 0.9], dtype=np.float64)
        pred_labels = np.array([1], dtype=np.int64)
        label_logits = np.array([-1.0, 1.0], dtype=np.float64)

        # Create and save ClassifierOutput
        original = ClassifierOutput(label_probs, pred_labels, label_logits)
        file_path = self.temp_path / "predictions.npz"
        original.to_npz(file_path)

        # Load and verify
        loaded = ClassifierOutput.from_npz(file_path)
        np.testing.assert_array_equal(loaded.label_probs, label_probs)
        np.testing.assert_array_equal(loaded.pred_labels, pred_labels)
        np.testing.assert_array_equal(loaded.label_logits, label_logits)

    def test_classifier_output_to_dict(self) -> None:
        label_probs = np.array([0.1, 0.9], dtype=np.float64)
        pred_labels = np.array([1], dtype=np.int64)
        label_logits = np.array([-1.0, 1.0], dtype=np.float64)
        output = ClassifierOutput(label_probs, pred_labels, label_logits)

        result = output.to_dict()
        self.assertIn("label_probs", result)
        self.assertIn("pred_labels", result)
        self.assertIn("label_logits", result)
        np.testing.assert_array_equal(result["label_probs"], label_probs)
        np.testing.assert_array_equal(result["pred_labels"], pred_labels)
        np.testing.assert_array_equal(result["label_logits"], label_logits)


class TestCalibratorOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_calibrator_output_save_load(self) -> None:
        label_probs = np.array([0.1, 0.9], dtype=np.float64)
        pred_labels = np.array([1], dtype=np.int64)

        original = CalibratorOutput(label_probs, pred_labels)
        file_path = self.temp_path / "calibrated_predictions.npz"
        original.to_npz(file_path)

        loaded = CalibratorOutput.from_npz(file_path)
        np.testing.assert_array_equal(loaded.label_probs, label_probs)
        np.testing.assert_array_equal(loaded.pred_labels, pred_labels)

    def test_calibrator_output_to_dict(self) -> None:
        label_probs = np.array([0.1, 0.9], dtype=np.float64)
        pred_labels = np.array([1], dtype=np.int64)
        output = CalibratorOutput(label_probs, pred_labels)

        result = output.to_dict()
        self.assertIn("label_probs", result)
        self.assertIn("pred_labels", result)
        self.assertNotIn("label_logits", result)
        np.testing.assert_array_equal(result["label_probs"], label_probs)
        np.testing.assert_array_equal(result["pred_labels"], pred_labels)


if __name__ == "__main__":
    unittest.main()
