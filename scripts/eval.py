import argparse
import json
import os

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from datasets import Dataset, load_dataset
from transformers import set_seed

from src.core.calibrators import BatchCalibrator, ContextFreeCalibrator, TemperatureCalibrator
from src.core.classifiers import GuardModel
from src.core.utils import compute_or_load_predictions
from src.evaluation.metrics import compute_metrics, print_summary


if TYPE_CHECKING:
    from src.core.calibrators import BaseCalibrator
    from src.core.types import CalibratorOutput

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

    true_labels = np.asarray([0 if x["is_safe"] else 1 for x in dataset.to_list()])

    # Initialize the guard model
    guard_model = GuardModel(args.model, taxonomy=args.taxonomy, descriptions=args.descriptions)

    print(SEPARATOR)
    print("Uncalibrated\n")

    model_kwargs = {"max_new_tokens": args.max_new_tokens}

    output = compute_or_load_predictions(
        guard_model,
        dataset,
        output_path / "evaluation" / "predictions.npz",
        model_kwargs=model_kwargs,
    )

    # Compute metrics for uncalibrated results
    metrics = {}
    metrics["uncalibrated"] = compute_metrics(true_labels, output, ece_bins=args.ece_bins)

    calibrators: dict[str, BaseCalibrator] = {
        "context-free": ContextFreeCalibrator(guard_model, token=["N/A"], model_kwargs=model_kwargs),
        "batch": BatchCalibrator(guard_model, output.label_probs, model_kwargs=model_kwargs),
        "temperature": TemperatureCalibrator(guard_model, temperature=5.6, model_kwargs=model_kwargs),
    }

    calibrated_outputs: dict[str, CalibratorOutput] = {}

    for method_name, calibrator in calibrators.items():
        print(SEPARATOR)
        print(f"Method: {method_name}\n")

        cal_output = calibrator.calibrate(output)

        calibrated_outputs[method_name] = cal_output
        metrics[method_name] = compute_metrics(true_labels, cal_output, ece_bins=args.ece_bins)

    print(SEPARATOR)

    # Save metrics
    with (output_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save calibrated results
    for method_name, cal_output in calibrated_outputs.items():
        cal_output.to_npz(output_path / "evaluation" / f"{method_name}_predictions.npz")

    print(SEPARATOR)
    print_summary(metrics)


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
    parser.add_argument("--output-path", type=str, default="results", help="Path to save the output")
    parser.add_argument("--show-plot", type=bool, default=True, help="Whether to plot the calibration curves")
    parser.add_argument("--plot-bins", type=int, default=15, help="Number of bins for calibration curve plotting")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum number of tokens to generate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
