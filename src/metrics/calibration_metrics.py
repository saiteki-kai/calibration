import numpy as np
import numpy.typing as npt
from netcal.metrics import ECE, MCE
from sklearn.metrics import (
    auc,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)


def compute_metrics(
    true_labels: npt.NDArray[np.int64],
    probs: npt.NDArray[np.float64],
    pred_labels: npt.NDArray[np.int64],
    ece_bins: int = 15
) -> dict[str, float]:
    # Create ECE calculator inside the function
    ece_calculator = ECE(bins=ece_bins)
    mce_calculator = MCE(bins=ece_bins)

    # Calculate calibration metrics
    ece_score = ece_calculator.measure(probs, true_labels)
    mce_score = mce_calculator.measure(probs, true_labels)
    brier_score = brier_score_loss(true_labels, pred_labels)
    print(f"ECE: {ece_score}")
    print(f"MCE: {mce_score}")
    print(f"Brier: {brier_score}")

    # Calculate classification metrics
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)

    print("F1: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Accuracy: ", accuracy)

    # Calculate AUPRC
    pr_precision, pr_recall, _ = precision_recall_curve(true_labels, pred_labels)
    auprc = auc(pr_recall, pr_precision)
    print(f"AUPRC: {auprc}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    return {
        "ece": ece_score,
        "brier": brier_score,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "auprc": auprc,
    }
