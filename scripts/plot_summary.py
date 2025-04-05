import argparse

from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import matplotlib.pyplot as plt
import numpy as np

from datasets import Dataset, load_dataset


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy import int64
    from numpy.typing import NDArray

from src.core.types import CalibratorOutput, ClassifierOutput, PredictionOutput
from src.evaluation.visualization import calibration_curve, confidence_histogram, reliability_diagram


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
        )

        reliability_diagram(
            true_labels=true_labels,
            pred_probs=uncalibrated_output.label_probs[:, 1],
            n_bins=n_bins,
            ax=axes[1, i],
        )

        axes[0, i].set_title(model)
        axes[1, i].set_title(model)

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
    outputs = {
        "uncalibrated_output": outputs["uncalibrated_output"],
        **cast("dict[str, PredictionOutput]", outputs["calibrated_outputs"]),
    }

    fig_name = f"Reliability Diagram and Confidence Distribution - {model_name}"
    fig, axes = plt.subplots(2, len(outputs), figsize=(20, 10), num=fig_name)

    for i, (method, output) in enumerate(outputs.items()):
        confidence_histogram(
            ax=axes[0, i],
            pred_probs=output.label_probs[:, 1],
            n_bins=n_bins * 2,
        )

        reliability_diagram(
            true_labels=true_labels,
            pred_probs=output.label_probs[:, 1],
            n_bins=n_bins,
            ax=axes[1, i],
        )

        axes[0, i].set_title(method)
        axes[1, i].set_title(method)

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
    outputs = {
        "uncalibrated_output": outputs["uncalibrated_output"],
        **cast("dict[str, PredictionOutput]", outputs["calibrated_outputs"]),
    }

    fig_name = f"Calibration Curve - {model_name}"
    fig, ax = plt.subplots(figsize=(20, 10), num=fig_name)

    pred_probs = {method: output.label_probs[:, 1] for (method, output) in outputs.items()}
    labels = list(outputs.keys())
    probs = list(pred_probs.values())

    calibration_curve(true_labels=true_labels, pred_probs=probs, ax=ax, n_bins=n_bins, label=labels)

    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, output_dir, f"{model_name}_calibration_curve")

    return fig


def load_data(
    dataset_name: str = "PKU-Alignment/Beavertails",
    split: str = "330k_test",
    sample_size: Optional[int] = None,
) -> tuple[Dataset, "NDArray[int64]"]:
    dataset = load_dataset(dataset_name, split=split)
    dataset = cast("Dataset", dataset)

    if sample_size is not None:
        dataset = dataset.select(range(sample_size))

    true_labels = np.asarray([0 if x["is_safe"] else 1 for x in dataset.to_list()])

    return dataset, true_labels


def get_model_name(model: str) -> str:
    return model.split("__")[-1] if "__" in model else model


def load_model_predictions(
    model: str,
    taxonomy: str,
    methods: list[str],
) -> tuple[str, ClassifierOutput, dict[str, CalibratorOutput]]:
    model_dir = Path(f"results/{model}/{taxonomy}")
    model_name = get_model_name(model)

    # Load uncalibrated predictions
    uncalibrated_output = ClassifierOutput.from_npz(model_dir / "evaluation" / "predictions.npz")

    # Load calibrated predictions
    calibrated_outputs = {}
    for method in methods:
        cal_output = CalibratorOutput.from_npz(model_dir / "evaluation" / f"{method}_predictions.npz")
        calibrated_outputs[method] = cal_output

    return model_name, uncalibrated_output, calibrated_outputs


def save_figure(fig: "Figure", path: Path | str, filename: str) -> None:
    filepath = Path(path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(filepath.with_suffix(".pdf")), dpi=300, bbox_inches="tight")
    fig.savefig(str(filepath.with_suffix(".png")), dpi=300, bbox_inches="tight")


def main(args: argparse.Namespace) -> None:
    models = ["meta-llama__Llama-Guard-3-1B", "meta-llama__Llama-Guard-3-1B"]
    taxonomy = "beavertails"
    sample_size = 10000
    plot_bins = 10

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    _, true_labels = load_data(dataset_name="PKU-Alignment/Beavertails", split="330k_test", sample_size=sample_size)

    outputs = {}
    for i, model in enumerate(models):
        model_name, uncalibrated_output, calibrated_outputs = load_model_predictions(
            model,
            taxonomy,
            methods=["context-free", "batch", "temperature"],
        )
        outputs[model_name + f"_{i}"] = {
            "uncalibrated_output": uncalibrated_output,
            "calibrated_outputs": calibrated_outputs,
        }

    compare_uncalibrated_model_reliability_diagram_and_confidence_histogram(
        outputs, true_labels, n_bins=plot_bins, output_dir=output_dir
    )

    for model_name, output in outputs.items():
        compare_calibrated_model_reliability_diagram(
            model_name,
            output,
            true_labels,
            n_bins=plot_bins,
            output_dir=output_dir,
        )

        compare_calibrated_model_curve(
            model_name,
            output,
            true_labels,
            n_bins=plot_bins,
            output_dir=output_dir,
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/comparison/plots")
    args = parser.parse_args()
    main(args)
