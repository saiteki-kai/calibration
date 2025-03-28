from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from tqdm import tqdm

from src.core.calibrators.calibration import calibrate_py
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray


class BaseCalibrator(ABC):
    _guard_model: GuardModel
    _calibration_mode: str
    _model_kwargs: dict[str, Any]

    def __init__(self, guard_model: GuardModel, model_kwargs: dict[str, Any] | None = None) -> None:
        self._guard_model = guard_model
        self._calibration_mode = "diagonal"
        self._model_kwargs = model_kwargs or {}

    def calibrate(self, probs: "NDArray[float64]") -> tuple["NDArray[float64]", "NDArray[int64]"]:
        prior = self._compute_prior()

        cal_probs = []
        cal_pred_labels = []

        for prob in tqdm(probs, desc="Calibrating predictions"):
            cal_prob = calibrate_py(prob, prior, mode=self._calibration_mode)
            cal_probs.append(cal_prob)
            pred_label = int(np.argmax(cal_prob.reshape(-1)))
            cal_pred_labels.append(pred_label)

        return (
            np.array(cal_probs).squeeze(),
            np.array(cal_pred_labels).squeeze(),
        )

    @abstractmethod
    def _compute_prior(self) -> "NDArray[float64]":
        msg = "Subclasses must implement compute_prior method"
        raise NotImplementedError(msg)
