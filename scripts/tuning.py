import argparse
import json

from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasets import Dataset, load_dataset
from sklearn.metrics import log_loss
from transformers import set_seed

from src.core.calibrators import BatchCalibrator, TemperatureCalibrator
from src.core.classifiers import GuardModel
from src.core.types import ClassifierOutput
from src.core.utils import compute_or_load_predictions
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualization.utils import save_figure


sns.set_theme(style="white", context="paper", font_scale=1.2)

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\\usepackage{libertinus}"


def tune_parameter(
    args: argparse.Namespace,
    guard_model: GuardModel,
    pred_output: ClassifierOutput,
    validation_set: Dataset,
    param_name: str,
    param_range: np.ndarray[float, np.dtype[np.float64]],
    metric: str,
    model_kwargs: dict[str, Any],
) -> tuple[float | None, dict[str, float], dict[str, list[float]]]:
    true_labels = np.asarray([0 if x["is_safe"] else 1 for x in validation_set.to_list()])

    lower_is_better = metric in ["nll", "ece", "mce"]

    best_param = None
    best_metric_value = float("inf") if lower_is_better else float("-inf")
    best_metrics = {}

    metrics_history = {
        param_name: [],
        "nll": [],
        "ece": [],
        "mce": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }

    if metric not in metrics_history:
        msg = f"Unknown metric: {metric}. Available metrics: {list(metrics_history.keys())}"
        raise ValueError(msg)

    for value in param_range:
        if param_name == "temperature":
            calibrator = TemperatureCalibrator(guard_model, temperature=value, model_kwargs=model_kwargs)
        # TODO: refactor passing the calibration method or the calibrator itself
        elif param_name == "gamma":
            calibrator = BatchCalibrator(guard_model, pred_output.label_probs, gamma=value, model_kwargs=model_kwargs)
        else:
            msg = f"Unknown parameter: {param_name}"
            raise ValueError(msg)

        cal_output = calibrator.calibrate(pred_output)
        metrics = compute_metrics(true_labels, cal_output, ece_bins=args.ece_bins)

        nll = log_loss(true_labels, cal_output.label_probs[:, 1])
        metrics["nll"] = float(nll)

        # Store metrics for plotting
        metrics_history[param_name].append(value)
        metrics_history["ece"].append(metrics["ece"])
        metrics_history["mce"].append(metrics["mce"])
        metrics_history["accuracy"].append(metrics["accuracy"])
        metrics_history["f1"].append(metrics["f1"])
        metrics_history["precision"].append(metrics["precision"])
        metrics_history["recall"].append(metrics["recall"])
        metrics_history["nll"].append(nll)

        # Check if this parameter value is the best so far
        if (lower_is_better and metrics[metric] < best_metric_value) or (
            not lower_is_better and metrics[metric] > best_metric_value
        ):
            best_metric_value = metrics[metric]
            best_param = value
            best_metrics = metrics

    return best_param, best_metrics, metrics_history


def plot_metrics(
    metrics_history: dict[str, list[float]], metric: str, param_name: str, output_path: Path, model_name: str = "Model"
) -> None:
    df_metrics = pd.DataFrame(metrics_history)
    best_idx = df_metrics[metric].idxmin() if metric in ["nll", "ece", "mce"] else df_metrics[metric].idxmax()
    best_param_value = df_metrics.loc[best_idx, param_name]

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    fig.suptitle(model_name, fontweight="bold")

    # NLL Plot
    sns.lineplot(data=df_metrics, x=param_name, y="nll", ax=axes[0], color="#1f77b4")
    axes[0].axvline(x=best_param_value, color="#d62728", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Negative Log-Likelihood")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Calibration Metrics Plot
    calibration_metrics = ["ece", "mce"]
    melted_cal = pd.melt(
        df_metrics, id_vars=[param_name], value_vars=calibration_metrics, var_name="Metric", value_name="Value"
    )

    sns.lineplot(
        data=melted_cal,
        x=param_name,
        y="Value",
        hue="Metric",
        style="Metric",
        dashes=False,
        palette=["#ff7f0e", "#2ca02c"],
        linewidth=1.5,
        markersize=8,
        ax=axes[1],
    )

    axes[1].axvline(x=best_param_value, color="#d62728", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Calibration Error")
    axes[1].legend(
        title="",
        ncol=len(calibration_metrics),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        frameon=False,
    )
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Classification Metrics Plot
    classification_metrics = ["accuracy", "f1", "precision", "recall"]
    melted_class = pd.melt(
        df_metrics, id_vars=[param_name], value_vars=classification_metrics, var_name="Metric", value_name="Value"
    )

    sns.lineplot(
        data=melted_class,
        x=param_name,
        y="Value",
        hue="Metric",
        style="Metric",
        dashes=False,
        palette=["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
        linewidth=1.5,
        markersize=8,
        ax=axes[2],
    )

    # Add best parameter vertical line
    axes[2].axvline(x=best_param_value, color="#d62728", linestyle="--", alpha=0.7)

    axes[2].set_xlabel(f"{param_name.capitalize()} Value")
    axes[2].set_ylabel("Classification Metrics")
    axes[2].legend(
        title="",
        ncol=len(classification_metrics),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        frameon=False,
    )
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Add scatter points at best parameter value
    idx = np.argmin(np.abs(np.array(df_metrics[param_name]) - best_param_value))
    axes[0].scatter([best_param_value], [df_metrics["nll"].iloc[idx]], color="#d62728", zorder=5, s=60)

    for metric_name in calibration_metrics:
        axes[1].scatter([best_param_value], [df_metrics[metric_name].iloc[idx]], color="#d62728", zorder=5, s=60)

    for metric_name in classification_metrics:
        axes[2].scatter([best_param_value], [df_metrics[metric_name].iloc[idx]], color="#d62728", zorder=5, s=60)

    # Add text annotation for the best value AFTER all plots are drawn
    best_metric = df_metrics[metric].min() if metric in ["nll", "ece", "mce"] else df_metrics[metric].max()
    ax_idx = 0 if metric == "nll" else 1 if metric in ["ece", "mce"] else 2

    # Calculate positioning for annotation
    param_range = df_metrics[param_name].to_numpy()
    param_min, param_max = param_range.min(), param_range.max()
    relative_pos = (best_param_value - param_min) / (param_max - param_min)
    param_span = param_range.max() - param_range.min()
    offset = param_span * 0.05

    if relative_pos > 0.6:
        text_x_data = best_param_value - offset
        ha = "right"
    else:
        text_x_data = best_param_value + offset
        ha = "left"

    # Get current axis limits and position annotation
    y_min, y_max = axes[ax_idx].get_ylim()

    if ax_idx == 0:  # NLL plot - align to top (lower is better)
        text_y_data = y_max - (y_max - y_min) * 0.08
        va = "top"
    else:  # Other plots - align to bottom (higher is better)
        text_y_data = y_min + (y_max - y_min) * 0.08
        va = "bottom"

    axes[ax_idx].annotate(
        f"Best {metric.capitalize() if len(metric) > 3 else metric.upper()}: {best_metric:.4f}",
        xy=(best_param_value, best_metric),
        xycoords="data",
        xytext=(text_x_data, text_y_data),
        textcoords="data",
        va=va,
        ha=ha,
        bbox={"boxstyle": "round,pad=0.3", "fc": "yellow", "alpha": 0.5},
        zorder=10,
    )

    fig.tight_layout()
    save_figure(fig, output_path, f"{param_name}_tuning_metrics")


def main(args: argparse.Namespace) -> None:
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Load the dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = cast("Dataset", dataset)

    if "train" in args.split and args.test_size > 0:
        validation_set = dataset.train_test_split(test_size=args.test_size, seed=args.seed)["test"]
    else:
        validation_set = dataset

    guard_model = GuardModel(args.model, taxonomy=args.taxonomy, descriptions=args.descriptions)
    model_kwargs = {"max_new_tokens": 10}

    # Ensure output path exists
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load or Compute Predictions
    pred_output = compute_or_load_predictions(
        guard_model,
        validation_set,
        args.output_path / "predictions" / "predictions.npz",
        model_kwargs=model_kwargs,
    )

    method_hyperparams = {
        "temperature": {
            "param": "temperature",
            "param_range": np.arange(0.1, 30.1, 0.1),  #  0.1
            "metric": "nll",
        },
        "batch": {
            "param": "gamma",
            "param_range": np.arange(-10, 10.1, 0.1),  # 0.1
            "metric": "accuracy",
        },
    }

    for method, hyperparams in method_hyperparams.items():
        print(f"Tuning {hyperparams['param'].capitalize()} for {method} method...")
        print(f"Parameter Range: {hyperparams['param_range']}")

        best_value, best_temp_metrics, temp_metrics_history = tune_parameter(
            args,
            guard_model,
            pred_output,
            validation_set,
            hyperparams["param"],
            hyperparams["param_range"],
            hyperparams["metric"],
            model_kwargs=model_kwargs,
        )

        plots_path = args.output_path / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
        plot_metrics(temp_metrics_history, hyperparams["metric"], hyperparams["param"], plots_path, args.model)

        print(f"Best {hyperparams['param'].capitalize()}: {best_value}")
        print(f"Best Metrics ({hyperparams['param'].capitalize()}): {best_temp_metrics}")

        with (args.output_path / f"{hyperparams['param']}.json").open("w", encoding="utf-8") as f:
            data = {"best": best_value, "best_metrics": best_temp_metrics, "history": temp_metrics_history}
            f.write(json.dumps(data, indent=4))

            print(f"{hyperparams['param'].capitalize()} tuning results saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune calibration parameters for Llama Guard 3")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-Guard-3-1B", help="Model to use")
    parser.add_argument("--dataset-name", type=str, default="PKU-Alignment/Beavertails", help="Dataset name")
    parser.add_argument("--split", type=str, default="330k_train", help="Split to use")
    parser.add_argument("--test-size", type=float, default=0.1, help="Size of the validation set")
    parser.add_argument("--taxonomy", type=str, default="llama-guard-3", help="Taxonomy used in chat template")
    parser.add_argument("--descriptions", type=bool, default=False, help="Whether to use safety descriptions")
    parser.add_argument("--ece-bins", type=int, default=15, help="Number of bins for ECE calculation")
    parser.add_argument("--output-path", type=Path, default="results", help="Path to save tuning results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
