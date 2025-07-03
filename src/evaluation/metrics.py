import logging

from typing import TYPE_CHECKING

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.core.types import PredictionOutput


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def compute_metrics(true_labels: "NDArray[int64]", preds: PredictionOutput, ece_bins: int = 15) -> dict[str, float]:
    # NOTE: works for binary classification only and assumes that the positive class is the second class (index 1)

    probs = preds.label_probs
    pred_labels = preds.pred_labels

    # Calculate calibration metrics
    ece_score = binary_ece(probs[:, 1], true_labels, ece_bins)
    mce_score = binary_mce(probs[:, 1], true_labels, ece_bins)
    brier_score = brier_score_loss(true_labels, probs[:, 1])
    nll = log_loss(true_labels, probs)

    # Calculate classification metrics
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)

    # Calculate AUPRC
    pr_precision, pr_recall, _ = precision_recall_curve(true_labels, probs[:, 1])
    auprc = auc(pr_recall, pr_precision)

    logger.info("Calibration Metrics:")
    logger.info("ECE: %s", ece_score)
    logger.info("MCE: %s", mce_score)
    logger.info("Brier: %s", brier_score)

    logger.info("Classification Metrics:")
    logger.info("F1: %s", f1)
    logger.info("Precision: %s", precision)
    logger.info("Recall: %s", recall)
    logger.info("Accuracy: %s", accuracy)
    logger.info("AUPRC: %s", auprc)

    logger.info("\nClassification Report:")
    logger.info(classification_report(true_labels, pred_labels))

    return {
        "ece": ece_score,
        "mce": mce_score,
        "brier": float(brier_score),
        "nll": float(nll),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auprc": float(auprc),
    }


def print_summary(metrics: dict[str, dict[str, float]]) -> None:
    metric_names = list(next(iter(metrics.values())).keys())

    col_width = max(len(name) for name in metric_names) + 2
    columns = "".join(f"{name:<{col_width}}" for name in metric_names)

    method_col_width = max(len(name) for name in metrics) + 4
    header = f"{'Method':<{method_col_width}}" + columns

    print(header)
    print("-" * (len(header) - (col_width - len(metric_names[-1]))))

    for method, metric in metrics.items():
        values = "".join(f"{metric[name]:<{col_width}.3f}" for name in metric_names)
        print(f"{method:{method_col_width}}{values}")


def get_bin_gaps(
    pred_probs: "NDArray[float64]", true_labels: "NDArray[int64]", n_bins: int = 15
) -> tuple["NDArray[float64]", "NDArray[float64]"]:
    """Compute bin gaps and weights for binary ECE and MCE calculation."""
    gaps = np.zeros(n_bins, dtype=np.float64)
    weights = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        bin_mask = (pred_probs > (i / n_bins if i > 0 else -1e-10)) & (pred_probs <= (i + 1) / n_bins)
        bin_samples = np.sum(bin_mask)

        if bin_samples > 0:
            mean_true = np.mean(true_labels[bin_mask])
            mean_conf = np.mean(pred_probs[bin_mask])

            gaps[i] = np.abs(mean_true - mean_conf)
            weights[i] = bin_samples / len(true_labels)

    return gaps, weights


def binary_ece(pred_probs: "NDArray[float64]", true_labels: "NDArray[int64]", n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification."""
    gaps, weights = get_bin_gaps(pred_probs, true_labels, n_bins)

    return float(sum(gaps * weights))


def binary_mce(pred_probs: "NDArray[float64]", true_labels: "NDArray[int64]", n_bins: int = 15) -> float:
    """Compute Maximum Calibration Error (MCE) for binary classification."""
    gaps, _ = get_bin_gaps(pred_probs, true_labels, n_bins)

    return float(np.max(gaps))
