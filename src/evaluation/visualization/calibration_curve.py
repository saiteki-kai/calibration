from typing import TYPE_CHECKING

from src.evaluation.visualization.utils import compute_calibration_curve


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy import float64, int64
    from numpy.typing import NDArray


def calibration_curve(
    ax: "Axes",
    true_labels: "NDArray[int64]",
    pred_probs: "NDArray[float64] | list[NDArray[float64]]",
    n_bins: int = 20,
    title: str = "Calibration Curve",
    color: str | tuple[float, float, float] | None = None,
    label: str | list[str] | None = None,
) -> None:
    if not isinstance(pred_probs, list):
        pred_probs = [pred_probs]

    if label is not None and not isinstance(label, list):
        label = [label]

    if label is not None and len(pred_probs) != len(label):
        msg = "pred_probs and label must have the same length"
        raise ValueError(msg)

    for i, prob_pred in enumerate(pred_probs):
        prob_true, prob_pred, _ = compute_calibration_curve(true_labels, prob_pred, n_bins=n_bins)

        ax.plot(
            prob_pred[prob_pred != -1],
            prob_true[prob_pred != -1],
            marker="s",
            linewidth=2,
            markersize=8,
            color=color,
            label=label[i] if label is not None else None,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, label="Perfect Calibration")

    ax.set_aspect("equal")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.set_title(title)
