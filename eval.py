from typing import cast

import numpy as np

from datasets import Dataset, load_dataset
from transformers import set_seed

from src.core.calibrator import GuardModelCalibrator
from src.core.classifiers.guard_model import GuardModel
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualization import plot_calibration_curves


def main() -> None:
    # Set random seed for reproducibility
    ece_bins = 15
    set_seed(42)

    # Load dataset
    dataset = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    dataset = cast("Dataset", dataset)

    # Initialize model
    guard_model = GuardModel("meta-llama/Llama-Guard-3-1B", "beavertails", descriptions=False)

    # Get uncalibrated predictions first
    pred_labels, label_probs = guard_model.predict(dataset.to_list())
    true_labels = np.asarray([0 if x["is_safe"] else 1 for x in dataset.to_list()])

    # Compute metrics for uncalibrated results
    print("Uncalibrated:")
    compute_metrics(true_labels, label_probs, pred_labels, ece_bins=ece_bins)

    # Save uncalibrated results
    uncalibrated_results = Dataset.from_dict(
        {
            "label_probs": label_probs,
            "pred_labels": pred_labels,
            "true_labels": true_labels,
        },
    )
    uncalibrated_results.to_json("uncalibrated_results.json")

    methods = ["batch", "context-free"]
    calibrated_results = []

    for method in methods:
        print(f"\nMethod: {method}")

        # Initialize calibrator
        calibrator = GuardModelCalibrator(guard_model, method=method)

        # Calibrate predictions using pre-computed probabilities
        print("Calibrating predictions...")
        cal_probs, cal_pred_labels = calibrator.calibrate(label_probs, pred_labels)
        calibrated_results.append((method, cal_probs))

        # Compute metrics for calibrated results
        print(f"\nCalibrated ({method}):")
        compute_metrics(true_labels, cal_probs, cal_pred_labels, ece_bins=ece_bins)

    # Plot all calibration curves in a single figure
    plot_calibration_curves(true_labels, label_probs, calibrated_results)


if __name__ == "__main__":
    main()
