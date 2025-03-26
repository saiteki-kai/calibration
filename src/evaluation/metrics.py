import logging

from typing import TYPE_CHECKING, cast

from netcal.metrics import ECE, MCE
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def compute_metrics(
    true_labels: "NDArray[int64]",
    probs: "NDArray[float64]",
    pred_labels: "NDArray[int64]",
    ece_bins: int = 15,
    verbose: bool = True,
) -> dict[str, float]:
    # Create ECE calculator inside the function
    ece_calculator = ECE(bins=ece_bins)
    mce_calculator = MCE(bins=ece_bins)

    # Calculate calibration metrics
    ece_score = cast("float", ece_calculator.measure(probs, true_labels))
    mce_score = cast("float", mce_calculator.measure(probs, true_labels))
    brier_score = brier_score_loss(true_labels, pred_labels)

    # Calculate classification metrics
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)

    # Calculate AUPRC
    pr_precision, pr_recall, _ = precision_recall_curve(true_labels, pred_labels)
    auprc = auc(pr_recall, pr_precision)

    if verbose:
        logger.info("ECE: %s", ece_score)
        logger.info("MCE: %s", mce_score)
        logger.info("Brier: %s", brier_score)
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
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auprc": float(auprc),
    }
