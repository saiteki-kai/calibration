from typing import TYPE_CHECKING, Any, override

import numpy as np

from src.core.calibrators.base import BaseCalibrator
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray


class TemperatureCalibrator(BaseCalibrator):
    def __init__(self, guard_model: GuardModel, temperature: float, model_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__(guard_model, model_kwargs)
        self.T = temperature

    @override
    def calibrate(self, logits: "NDArray[float64]") -> tuple["NDArray[float64]", "NDArray[int64]"]:
        calibrated_probs = np.array([np.exp(logit / self.T) / np.sum(np.exp(logit / self.T)) for logit in logits])
        pred_labels = np.argmax(calibrated_probs, axis=1)

        return calibrated_probs, pred_labels

    def _compute_prior(self) -> "NDArray[float64]":
        return np.empty((), dtype=np.float64)
