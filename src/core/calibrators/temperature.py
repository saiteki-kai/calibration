from typing import TYPE_CHECKING, Any, override

import numpy as np

from src.core.calibrators.base import BaseCalibrator
from src.core.classifiers.guard_model import GuardModel
from src.core.types import CalibratorOutput, ClassifierOutput


if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray


class TemperatureCalibrator(BaseCalibrator):
    def __init__(self, guard_model: GuardModel, temperature: float, model_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__(guard_model, model_kwargs)

        if temperature <= 0:
            msg = "Temperature must be positive, got {0}".format(temperature)
            raise ValueError(msg)

        self.T = temperature

    @override
    def calibrate(self, preds: ClassifierOutput) -> CalibratorOutput:
        calibrated_probs = softmax(preds.label_logits / self.T, axis=-1)
        pred_labels = np.argmax(calibrated_probs, axis=1)

        return CalibratorOutput(label_probs=calibrated_probs, pred_labels=pred_labels)

    def _calibrate_prob(self, _prob: "NDArray[float64]", _prior: "NDArray[float64]") -> "NDArray[float64]":
        return np.empty((), dtype=np.float64)

    def _compute_prior(self) -> "NDArray[float64]":
        return np.empty((), dtype=np.float64)


def softmax(x: "NDArray[float64]", axis: int = -1) -> "NDArray[float64]":
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # avoid overflow

    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
