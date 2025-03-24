from typing import Literal

import numpy as np

from .calibrators.context_free import ContextFreeCalibrator
from .calibrators.batch import BatchCalibrator
from .models.guard_model import GuardModel


class GuardModelCalibrator:
    def __init__(self, guard_model: GuardModel, method: Literal["context-free", "batch"]):
        self.guard_model = guard_model
        self.method = method

        calibrators = {
            "context-free": ContextFreeCalibrator,
            "batch": BatchCalibrator,
        }

        if method not in calibrators:
            raise ValueError(f"Unknown calibration method: {method}. Available methods: {list(calibrators.keys())}")

        self.calibrator = calibrators[method](guard_model)

    def predict(self, data):
        return self.guard_model.predict(data)

    def _format_predictions(self, preds):
        if isinstance(preds, tuple):
            probs, pred_labels = preds
        elif isinstance(preds, dict):
            probs = preds["label_probs"].cpu().numpy()
            pred_labels = np.array([preds["pred_label"]])
        else:
            probs = np.array([p["label_probs"].cpu().numpy() for p in preds])
            pred_labels = np.array([p["pred_label"] for p in preds])

        return probs, pred_labels

    def _format_results(self, calibrated_probs, calibrated_pred_labels):
        results = []
        for i in range(len(calibrated_probs)):
            results.append(
                {
                    "label_probs": calibrated_probs[i],
                    "pred_label": int(calibrated_pred_labels[i]),
                }
            )

        return results[0] if len(results) == 1 else results

    def calibrate(self, data):
        preds = self.predict(data)
        probs, pred_labels = self._format_predictions(preds)
        calibrated_probs, calibrated_pred_labels = self.calibrator.calibrate(probs, pred_labels)

        return self._format_results(calibrated_probs, calibrated_pred_labels)

    def calibrate_predictions(self, probs, pred_labels):
        return self.calibrator.calibrate(probs, pred_labels)

    def compute_prior(self, precomputed_probs=None):
        return self.calibrator.compute_prior(precomputed_probs)
