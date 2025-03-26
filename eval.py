import argparse
import json
import os

from pathlib import Path
from typing import cast

import numpy as np

from datasets import Dataset, load_dataset
from transformers import set_seed

from src.core.calibrator import GuardModelCalibrator
from src.core.classifiers.guard_model import GuardModel
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualization import plot_calibration_curves


SEPARATOR = "-" * os.get_terminal_size().columns


def main(args: argparse.Namespace) -> None:
    # Set random seed for reproducibility
    set_seed(args.seed)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = cast("Dataset", dataset)

    if args.sample_size is not None:
        dataset = dataset.select(range(args.sample_size))

    # Initialize model
    guard_model = GuardModel(args.model, taxonomy=args.taxonomy, descriptions=args.descriptions)

    # Get uncalibrated predictions first
    print(SEPARATOR)
    print("Uncalibrated\n")
    pred_labels, label_probs = guard_model.predict(dataset.to_list())
    true_labels = np.asarray([0 if x["is_safe"] else 1 for x in dataset.to_list()])

    # Compute metrics for uncalibrated results
    metrics = {}
    metrics["uncalibrated"] = compute_metrics(
        true_labels,
        label_probs,
        pred_labels,
        ece_bins=args.ece_bins,
        verbose=args.verbose,
    )

    # Save uncalibrated results
    predictions = Dataset.from_dict(
        {
            "label_probs": label_probs,
            "pred_labels": pred_labels,
            "true_labels": true_labels,
        },
    )
    predictions.to_json(output_path / "predictions.json")

    methods = ["batch", "context-free"]
    calibrated_results = []

    for method in methods:
        print(SEPARATOR)
        print(f"Method: {method}\n")

        # Initialize calibrator
        calibrator = GuardModelCalibrator(guard_model, method=method)

        # Calibrate predictions using pre-computed probabilities
        cal_probs, cal_pred_labels = calibrator.calibrate(label_probs, pred_labels)
        calibrated_results.append((method, cal_probs, cal_pred_labels))

        # Compute metrics for calibrated results
        metrics[method] = compute_metrics(
            true_labels,
            cal_probs,
            cal_pred_labels,
            ece_bins=args.ece_bins,
            verbose=args.verbose,
        )

    print(SEPARATOR)

    # Save metrics
    with (output_path / "metrics.json").open("w") as f:
        json.dump(metrics, f)

    # Save calibrated results
    calibrated_predictions = Dataset.from_dict(
        dict(zip(["method", "calibrated_probs", "calibrated_labels"], zip(*calibrated_results)))
    )
    calibrated_predictions.to_json(output_path / "calibrated_predictions.json")

    print(SEPARATOR)
    print_metrics_summary(metrics)

    print(SEPARATOR)

    # Plot all calibration curves in a single figure
    plot_calibration_curves(
        true_labels,
        label_probs,
        calibrated_results,
        output_path=output_path / "plots",
        show_plot=args.show_plot,
        n_bins=args.plot_bins,
        title=args.model + "\n" + args.dataset_name,
    )


def print_metrics_summary(metrics: dict[str, dict[str, float]]) -> None:
    print("Metrics Comparison Summary\n")

    metric_names = list(next(iter(metrics.values())).keys())

    col_width = 12
    columns = "".join("{:<{col_width}}".format(name, col_width=col_width) for name in metric_names)
    header = "{:<15}".format("Method") + columns

    print(header)
    print("-" * (len(header) - (col_width - len(metric_names[-1]))))

    for method, metric in metrics.items():
        values = "".join("{:<{col_width}.3f}".format(metric[name], col_width=col_width) for name in metric_names)
        print(f"{method:<15}{values}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate Llama Guard 3 predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-Guard-3-1B", help="Model to use")
    parser.add_argument("--dataset-name", type=str, default="PKU-Alignment/Beavertails", help="Dataset name")
    parser.add_argument("--split", type=str, default="330k_test", help="Split to use")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of samples to use from the dataset")
    parser.add_argument("--taxonomy", type=str, default="llama-guard-3", help="Taxonomy used in chat template")
    parser.add_argument("--descriptions", type=bool, default=False, help="Whether to use safety descriptions")
    parser.add_argument("--ece-bins", type=int, default=15, help="Number of bins for ECE calculation")
    parser.add_argument("--verbose", type=bool, default=True, help="Whether to print verbose output")
    parser.add_argument("--output-path", type=str, default="results", help="Path to save the output")
    parser.add_argument("--show-plot", type=bool, default=True, help="Whether to plot the calibration curves")
    parser.add_argument("--plot-bins", type=int, default=20, help="Number of bins for calibration curve plotting")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
