from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from tqdm import tqdm

from src.core.calibrators.calibration import calibrate_py
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray


class BaseCalibrator(ABC):
    def __init__(self, guard_model: GuardModel) -> None:
        self.guard_model = guard_model
        self.calibration_mode = "diagonal"

    @abstractmethod
    def compute_prior(self) -> "NDArray[float64]":
        msg = "Subclasses must implement compute_prior method"
        raise NotImplementedError(msg)

    def calibrate(
        self,
        pred_probs: "NDArray[float64]",
        _pred_labels: "NDArray[int64]",
    ) -> tuple["NDArray[float64]", "NDArray[int64]"]:
        prior = self.compute_prior()

        calibrated_probs = []
        calibrated_pred_labels = []

        for prob in tqdm(pred_probs, desc="Calibrating predictions"):
            cal_prob = calibrate_py(prob, prior, mode=self.calibration_mode)
            calibrated_probs.append(cal_prob)
            pred_label = int(np.argmax(cal_prob.reshape(-1)))
            calibrated_pred_labels.append(pred_label)

        return (
            np.array(calibrated_probs).squeeze(),
            np.array(calibrated_pred_labels).squeeze(),
        )
