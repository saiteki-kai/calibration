import matplotlib.pyplot as plt
import numpy as np


def confidence_histogram(
    ax: plt.Axes,
    pred_probs: np.ndarray,
    n_bins: int = 10,
    title: str = "Confidence Distribution",
) -> None:
    counts, bins = np.histogram(pred_probs, bins=n_bins)
    counts = counts / sum(counts)

    ax.hist(bins[:-1], bins, weights=counts, linewidth=1.5, edgecolor="#0054a7", alpha=0.85, color="#007af3")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax.set_xlabel("Confidence")
    ax.set_ylabel(r"\% of Samples")
    ax.set_title(title)
