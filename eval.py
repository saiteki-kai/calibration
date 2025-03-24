import numpy as np
from datasets import load_dataset
from transformers import set_seed

from src.guard_calibrator.calibrator import GuardModelCalibrator
from src.guard_calibrator.models.guard_model import GuardModel
from src.visualization.plotting import plot_calibration_curves
from src.metrics.calibration_metrics import compute_metrics


def main():
    # Set random seed for reproducibility
    ece_bins = 15
    set_seed(42)

    # Load dataset
    dataset = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    dataset = dataset.select(range(2000))  # Use a small subset for testing

    # Initialize model and calibrator
    guard_model = GuardModel("meta-llama/Llama-Guard-3-1B", "llama-guard-3")

    # Get uncalibrated predictions first
    pred_labels, label_probs = guard_model.predict(dataset)
    true_labels = np.asarray([0 if x["is_safe"] else 1 for x in dataset])

    # Compute metrics for uncalibrated results
    print("Uncalibrated:")
    compute_metrics(true_labels, label_probs, pred_labels, ece_bins=ece_bins)

    methods = ["batch", "context-free"]
    cal_results = []

    for method in methods:
        print(f"\nMethod: {method}")

        # Initialize calibrator
        calibrator = GuardModelCalibrator(guard_model, method=method)

        # Calibrate predictions using pre-computed probabilities
        print("Calibrating predictions...")
        cal_probs, cal_pred_labels = calibrator.calibrate_predictions(label_probs, pred_labels)
        cal_results.append((method, cal_probs))

        # Compute metrics for calibrated results
        print(f"\nCalibrated ({method}):")
        compute_metrics(true_labels, cal_probs, cal_pred_labels, ece_bins=ece_bins)

    # Plot all calibration curves in a single figure
    plot_calibration_curves(true_labels, label_probs, cal_results)


if __name__ == "__main__":
    main()
