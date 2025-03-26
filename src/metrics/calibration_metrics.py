import logging

from typing import cast

import numpy as np
import numpy.typing as npt

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


logger = logging.getLogger(__name__)


def compute_metrics(
    true_labels: npt.NDArray[np.int64],
    probs: npt.NDArray[np.float64],
    pred_labels: npt.NDArray[np.int64],
    ece_bins: int = 15,
    verbose: bool = True,
) -> dict[str, float]:
    # Create ECE calculator inside the function
    ece_calculator = ECE(bins=ece_bins)
    mce_calculator = MCE(bins=ece_bins)

    # Calculate calibration metrics
    ece_score = cast(float, ece_calculator.measure(probs, true_labels))
    mce_score = cast(float, mce_calculator.measure(probs, true_labels))
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
        logger.info(f"ECE: {ece_score}")
        logger.info(f"MCE: {mce_score}")
        logger.info(f"Brier: {brier_score}")
        logger.info(f"F1: {f1}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"AUPRC: {auprc}")
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
