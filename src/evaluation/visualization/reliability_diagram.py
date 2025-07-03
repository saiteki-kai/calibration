from typing import TYPE_CHECKING

from src.evaluation.metrics import binary_ece, binary_mce
from src.evaluation.visualization.utils import compute_calibration_curve


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy import float64, int64
    from numpy.typing import NDArray


def reliability_diagram(
    ax: "Axes",
    true_labels: "NDArray[int64]",
    pred_probs: "NDArray[float64]",
    n_bins: int = 10,
    ece_bins: int = 15,
    title: str = "Reliability Diagram",
) -> None:
    prob_true, prob_pred, bins = compute_calibration_curve(true_labels, pred_probs, n_bins=n_bins)

    # Empty bins
    prob_true[prob_true == -1] = 0
    prob_pred[prob_pred == -1] = 0

    bin_width = 1 / n_bins
    linewidth = 1.5

    # ax.grid(True, linestyle="-", alpha=0.6)

    ax.bar(
        bins,
        prob_true,
        width=bin_width,
        align="edge",
        color="#007af3",
        edgecolor="#0054a7",
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
        edgecolor="#ff4040",
        color="#ff4040",
        alpha=0.4,
        linewidth=linewidth,
        hatch="\\",
    )

    ax.plot([0, 1], [0, 1], color="black", alpha=0.5, linestyle="--", label="Perfect Calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")

    ax.set_title(title)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("Confidence")
    ax.legend(loc="upper left")

    ece_value = binary_ece(pred_probs, true_labels, ece_bins)
    mce_value = binary_mce(pred_probs, true_labels, ece_bins)

    ax.text(
        0.97,
        0.03,
        f"ECE: {ece_value:.3f}\nMCE: {mce_value:.3f}",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "lightgrey",
            "alpha": 0.8,
        },
    )
