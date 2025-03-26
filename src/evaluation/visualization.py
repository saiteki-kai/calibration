from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from sklearn.calibration import calibration_curve


if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_calibration_curves(
    true_labels: npt.NDArray[np.int64],
    label_probs: npt.NDArray[np.float64],
    cal_results: list[tuple[str, npt.NDArray[np.float64]]],
    n_bins: int = 20,
    output_path: Path | None = None,
    show_plot: bool = True,
) -> "Figure":
    sns.set_style("white")
    sns.set_context("paper", font_scale=2)

    fig, ax = plt.subplots(figsize=(12, 8), num="Calibration Curves")

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfect Calibration")

    # Uncalibrated model line
    prob_true, prob_pred = calibration_curve(true_labels, label_probs[:, 1], n_bins=n_bins)
    sns.lineplot(x=prob_pred, y=prob_true, ax=ax, marker="s", label="Uncalibrated", linewidth=2, markersize=8)

    # Calibrated model lines
    colors = sns.color_palette("husl", len(cal_results))

    for (method_name, cal_probs), color in zip(cal_results, colors):
        prob_true, prob_pred = calibration_curve(true_labels, cal_probs[:, 1], n_bins=n_bins)

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

    ax.set_xlabel("Predicted Probability", labelpad=10)
    ax.set_ylabel("True Probability", labelpad=10)
    ax.set_title("Calibration Curves Comparison")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig
