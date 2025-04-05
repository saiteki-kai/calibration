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

from src.core.types import PredictionOutput


if TYPE_CHECKING:
    from numpy import int64
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def compute_metrics(true_labels: "NDArray[int64]", preds: PredictionOutput, ece_bins: int = 15) -> dict[str, float]:
    # NOTE: works for binary classification only and assumes that the positive class is the second class (index 1)

    probs = preds.label_probs
    pred_labels = preds.pred_labels

    # Create ECE calculator inside the function
    ece_calculator = ECE(bins=ece_bins)
    mce_calculator = MCE(bins=ece_bins)

    # Calculate calibration metrics
    ece_score = cast("float", ece_calculator.measure(probs, true_labels))
    mce_score = cast("float", mce_calculator.measure(probs, true_labels))
    brier_score = brier_score_loss(true_labels, probs[:, 1])

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
