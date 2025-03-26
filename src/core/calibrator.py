from typing import TYPE_CHECKING, Any

from src.core.calibrators.batch import BatchCalibrator
from src.core.calibrators.context_free import ContextFreeCalibrator
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray

    from src.core.calibrators.base import BaseCalibrator


class GuardModelCalibrator:
    def __init__(self, guard_model: GuardModel, method: str) -> None:
        self.guard_model = guard_model
        self.method = method

        calibrators: dict[str, type["BaseCalibrator"]] = {
            "context-free": ContextFreeCalibrator,
            "batch": BatchCalibrator,
        }

        if method not in calibrators:
            msg = f"Unknown calibration method: {method}. Available methods: {list(calibrators.keys())}"
            raise ValueError(msg)

        self.calibrator = calibrators[method](guard_model)

    def predict_and_calibrate(self, data: list[dict[str, str]]) -> list[dict[str, Any]]:
        pred_labels, pred_probs = self.guard_model.predict(data)
        calibrated_probs, calibrated_pred_labels = self.calibrator.calibrate(pred_probs, pred_labels)

        return self._format_results(calibrated_probs, calibrated_pred_labels)

    def calibrate(
        self,
        probs: "NDArray[float64]",
        pred_labels: "NDArray[int64]",
    ) -> tuple["NDArray[float64]", "NDArray[int64]"]:
        return self.calibrator.calibrate(probs, pred_labels)

    def compute_prior(self, precomputed_probs: "NDArray[float64] | None" = None) -> "NDArray[float64]":
        return self.calibrator.compute_prior(precomputed_probs)

    def _format_results(
        self,
        calibrated_probs: "NDArray[float64]",
        calibrated_pred_labels: "NDArray[int64]",
    ) -> list[dict[str, Any]]:
        return [
            {
                "label_probs": calibrated_probs[i],
                "pred_label": int(calibrated_pred_labels[i]),
            }
            for i in range(len(calibrated_probs))
        ]
