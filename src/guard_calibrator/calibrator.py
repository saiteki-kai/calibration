"""Main calibrator class."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from datasets import Dataset

from .calibrators.context_free import ContextFreeCalibrator
from .calibrators.batch import BatchCalibrator
from .models.guard_model import GuardModel


class GuardModelCalibrator:
    def __init__(self, guard_model: GuardModel, method: Literal["context-free", "batch"]):
        self.guard_model = guard_model

        # Initialize the appropriate calibrator based on method
        if method == "context-free":
            self.calibrator = ContextFreeCalibrator(guard_model)
        elif method == "batch":
            self.calibrator = BatchCalibrator(guard_model)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def predict(
        self,
        data: dict[str, str] | list[dict[str, str]] | Dataset,
    ) -> dict[str, float | int] | list[dict[str, float | int]] | tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Get raw predictions from the guard model."""
        return self.guard_model.predict(data)

    def calibrate(
        self,
        data: dict[str, str] | list[dict[str, str]] | Dataset,
    ) -> dict[str, float | int] | list[dict[str, float | int]]:
        # Get raw predictions
        preds = self.predict(data)

        # Convert to list format for processing
        if isinstance(preds, tuple):
            probs, pred_labels = preds
        elif isinstance(preds, dict):
            probs = preds["label_probs"].cpu().numpy()
            pred_labels = np.array([preds["pred_label"]])
        else:
            probs = np.array([p["label_probs"].cpu().numpy() for p in preds])
            pred_labels = np.array([p["pred_label"] for p in preds])

        calibrated_probs, calibrated_pred_labels = self.calibrator.calibrate(probs, pred_labels)

        results = []
        for i in range(len(calibrated_probs)):
            results.append(
                {
                    "label_probs": calibrated_probs[i],
                    "pred_label": int(calibrated_pred_labels[i]),
                }
            )

        return results[0] if len(results) == 1 else results

    def calibrate_predictions(
        self,
        probs: npt.NDArray[np.float64],
        pred_labels: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        return self.calibrator.calibrate(probs, pred_labels)

    def compute_prior(self, precomputed_probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        return self.calibrator.compute_prior(precomputed_probs)
