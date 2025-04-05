from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy import float64, int64
    from numpy.typing import NDArray


def compute_calibration_curve(
    y_true: "NDArray[int64]",
    y_pred: "NDArray[float64]",
    n_bins: int = 15,
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


def save_figure(fig: "Figure", path: Path | str, filename: str) -> None:
    filepath = Path(path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(filepath.with_suffix(".pdf")), dpi=300, bbox_inches="tight")
    fig.savefig(str(filepath.with_suffix(".png")), dpi=300, bbox_inches="tight")
