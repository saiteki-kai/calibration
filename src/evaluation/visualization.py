from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

from src.core.types import PredictionOutput


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy import float64, int64
    from numpy.typing import NDArray


def plot_calibration_curves(
    true_labels: "NDArray[int64]",
    label_probs: "NDArray[float64]",
    cal_results: dict[str, PredictionOutput],
    n_bins: int = 20,
    output_path: Path | str | None = None,
    show_plot: bool = True,
) -> "Figure":
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(8, 8), num="Calibration Curves")

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfect Calibration")

    # Uncalibrated model line
    prob_true, prob_pred, _ = calibration_curve(true_labels, label_probs[:, 1], n_bins=n_bins)
    sns.lineplot(x=prob_pred, y=prob_true, ax=ax, marker="s", label="Uncalibrated", linewidth=2, markersize=8)

    # Calibrated model lines
    colors = sns.color_palette("husl", len(cal_results))

    for (method_name, cal_output), color in zip(cal_results.items(), colors):
        prob_true, prob_pred, _ = calibration_curve(true_labels, cal_output.label_probs[:, 1], n_bins=n_bins)

        sns.lineplot(
            x=prob_pred,
            y=prob_true,
            ax=ax,
            marker="s",
            label=f"Calibrated ({method_name})",
            linewidth=2,
            markersize=8,
            color=color,
        )

    ax.set_xlabel("Confidence", labelpad=10)
    ax.set_ylabel("Accuracy", labelpad=10)

    ax.set_aspect("equal")
    ax.legend(loc="lower right")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path / "calibration_curves.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(output_path / "calibration_curves.png", dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig


def plot_reliability_diagram(
    bins: "NDArray[float64]",
    prob_true: "NDArray[float64]",
    prob_pred: "NDArray[float64]",
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
    show_plot: bool = True,
) -> "Figure":
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    bin_width = 1 / (len(bins) - 1)

    output_color = "#007af3"
    output_edge = "#0054a7"
    gap_color = "#ff4040"
    gap_edge = "#f30000"
    gap_alpha = 0.4
    gap_hatch = "\\"
    linewidth = 1.5

    ax.bar(
        bins,
        prob_true,
        width=bin_width,
        align="edge",
        color=output_color,
        edgecolor=output_edge,
        linewidth=linewidth,
        alpha=0.85,
        label="Outputs",
    )
    ax.bar(
        bins,
        prob_pred - prob_true,
        width=bin_width,
        bottom=prob_true,
        align="edge",
        label="Gap",
        edgecolor=gap_edge,
        color=gap_color,
        alpha=gap_alpha,
        linewidth=linewidth,
        hatch=gap_hatch,
    )

    ax.plot([0, 1], [0, 1], color="#444444", linestyle="--", label="Perfect Calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="-", alpha=0.6)

    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Reliability Diagram")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Confidence")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path / "reliability_diagram.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(output_path / "reliability_diagram.png", dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig


def calibration_curve(
    y_true: "NDArray[int64]", y_pred: "NDArray[float64]", n_bins: int = 20
) -> tuple["NDArray[float64]", "NDArray[float64]", "NDArray[float64]"]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_pred)

    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = np.zeros(len(bin_total))
    prob_true[nonzero] = bin_true[nonzero] / bin_total[nonzero]

    prob_pred = np.zeros(len(bin_total))
    prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred, bins


if __name__ == "__main__":
    import numpy as np

    n_bins = 5
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.5, 0.3, 0.2, 1.0])

    prob_true, prob_pred, bins = calibration_curve(y_true, y_pred, n_bins=n_bins)
    plot_reliability_diagram(bins, prob_true, prob_pred)
