import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve


def plot_calibration_curves(true_labels, label_probs, cal_results, n_bins=20):
    sns.set_style("white")
    sns.set_context("paper", font_scale=2)

    fig, ax = plt.subplots(figsize=(12, 8), num="Calibration Curves")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfect Calibration")

    prob_true, prob_pred = calibration_curve(true_labels, label_probs[:, 1], n_bins=n_bins)
    sns.lineplot(x=prob_pred, y=prob_true, ax=ax, marker="s", label="Uncalibrated", linewidth=2, markersize=8)

    colors = sns.color_palette("husl", 2)  # Use seaborn's color palette

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

    plt.tight_layout()
    plt.savefig("calibration_curves.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.show()
