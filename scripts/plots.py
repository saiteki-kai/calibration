import argparse

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

from datasets import Dataset, load_dataset

from src.core.types import CalibratorOutput, ClassifierOutput, PredictionOutput
from src.evaluation.visualization import calibration_curve, confidence_histogram, reliability_diagram
from src.evaluation.visualization.utils import save_figure


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy import int64
    from numpy.typing import NDArray

METHODS_NAMES = {
    "uncalibrated_output": "Uncalibrated",
    "batch": "Batch Calibration",
    "context-free": "Context-free Calibration",
    "temperature": "Temperature Scaling",
}


def compare_uncalibrated_model_reliability_diagram_and_confidence_histogram(
    outputs: dict[str, dict[str, PredictionOutput]],
    true_labels: "NDArray[int64]",
    n_bins: int,
    output_dir: Path | str | None = None,
) -> "Figure":
    fig_name = "Reliability Diagram and Confidence Distribution - Uncalibrated Models"
    fig, axes = plt.subplots(2, len(outputs), figsize=(10, 10), num=fig_name)

    for i, (model, output) in enumerate(outputs.items()):
        uncalibrated_output = output["uncalibrated_output"]

        confidence_histogram(
            ax=axes[0, i],
            pred_probs=uncalibrated_output.label_probs[:, 1],
            n_bins=n_bins * 2,
            title=model,
        )

        reliability_diagram(
            ax=axes[1, i],
            true_labels=true_labels,
            pred_probs=uncalibrated_output.label_probs[:, 1],
            n_bins=n_bins,
            title="",
        )

    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, "uncalibrated_reliability_diagram")

    return fig


def compare_calibrated_model_reliability_diagram(
    model_name: str,
    outputs: dict[str, PredictionOutput],
    true_labels: "NDArray[int64]",
    n_bins: int,
    output_dir: Path | str | None = None,
) -> "Figure":
    fig_name = f"Reliability Diagram and Confidence Distribution - {model_name}"
    fig, axes = plt.subplots(2, len(outputs), figsize=(20, 10), num=fig_name)

    for i, (method, output) in enumerate(outputs.items()):
        confidence_histogram(
            ax=axes[0, i],
            pred_probs=output.label_probs[:, 1],
            n_bins=n_bins * 2,
            title=METHODS_NAMES[method],
        )

        reliability_diagram(
            true_labels=true_labels,
            pred_probs=output.label_probs[:, 1],
            n_bins=n_bins,
            ax=axes[1, i],
            title="",
        )

    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, f"{model_name}_reliability_diagram")

    return fig


def compare_calibrated_model_curve(
    model_name: str,
    outputs: dict[str, PredictionOutput],
    true_labels: "NDArray[int64]",
    n_bins: int,
    output_dir: Path | str | None = None,
) -> "Figure":
    fig_name = f"Calibration Curve - {model_name}"
    fig, ax = plt.subplots(figsize=(20, 10), num=fig_name)

    pred_probs = {METHODS_NAMES[method]: output.label_probs[:, 1] for (method, output) in outputs.items()}
    labels = list(pred_probs.keys())
    probs = list(pred_probs.values())

    calibration_curve(true_labels=true_labels, pred_probs=probs, ax=ax, n_bins=n_bins, label=labels)

    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, f"{model_name}_calibration_curve")

    return fig


def load_model_predictions(
    model_name: str,
    taxonomy: str,
    methods: list[str],
) -> tuple[str, ClassifierOutput, dict[str, CalibratorOutput]]:
    model_dir = Path(f"results/{model_name.replace('/', '__')}/{taxonomy}")

    # Load uncalibrated predictions
    uncalibrated_output = ClassifierOutput.from_npz(model_dir / "evaluation" / "predictions.npz")

    # Load calibrated predictions
    calibrated_outputs = {}
    for method in methods:
        cal_output = CalibratorOutput.from_npz(model_dir / "evaluation" / f"{method}_predictions.npz")
        calibrated_outputs[method] = cal_output

    return model_name, uncalibrated_output, calibrated_outputs


def main(args: argparse.Namespace) -> None:
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = cast("Dataset", dataset)

    if args.sample_size is not None:
        dataset = dataset.select(range(args.sample_size))

    true_labels = np.asarray([0 if x["is_safe"] else 1 for x in dataset.to_list()])

    outputs = {}
    for i, model in enumerate(args.models):
        model_name, uncalibrated_output, calibrated_outputs = load_model_predictions(
            model,
            args.taxonomy,
            methods=["context-free", "batch", "temperature"],
        )
        outputs[model_name] = {"uncalibrated_output": uncalibrated_output, **calibrated_outputs}

    compare_uncalibrated_model_reliability_diagram_and_confidence_histogram(
        outputs, true_labels, n_bins=args.plot_bins, output_dir=output_dir
    )

    for model_name, output in outputs.items():
        compare_calibrated_model_reliability_diagram(
            model_name,
            output,
            true_labels,
            n_bins=args.plot_bins,
            output_dir=output_dir,
        )

        compare_calibrated_model_curve(
            model_name,
            output,
            true_labels,
            n_bins=args.plot_bins,
            output_dir=output_dir,
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", help="List of model names")
    parser.add_argument("--dataset_name", type=str, default="PKU-Alignment/Beavertails", help="Name of the dataset")
    parser.add_argument("--split", type=str, default="330k_test", help="Dataset split to use")
    parser.add_argument("--taxonomy", type=str, default="beavertails", help="Taxonomy to use")
    parser.add_argument("--sample_size", type=int, default=2000, help="Number of samples to use")
    parser.add_argument("--plot_bins", type=int, default=10, help="Number of bins for plots")
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/comparison/plots")
    args = parser.parse_args()
    main(args)
