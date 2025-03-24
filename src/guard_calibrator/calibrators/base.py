from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..models.guard_model import GuardModel
from ..utils.calibration import calibrate_py


class BaseCalibrator(ABC):
    def __init__(self, guard_model: GuardModel):
        self.guard_model = guard_model

    @abstractmethod
    def compute_prior(self, probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Subclasses must implement compute_prior method")

    @abstractmethod
    def get_calibration_mode(self) -> Literal["diagonal", "identity"]:
        raise NotImplementedError("Subclasses must implement get_calibration_mode method")

    def calibrate(
        self,
        pred_probs: npt.NDArray[np.float64],
        pred_labels: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        prior = self.compute_prior(pred_probs)
        mode = self.get_calibration_mode()

        calibrated_probs = []
        calibrated_pred_labels = []

        for prob in pred_probs:
            cal_prob = calibrate_py(prob, prior, mode=mode)
            calibrated_probs.append(cal_prob)
            pred_label = int(np.argmax(cal_prob.reshape(-1)))
            calibrated_pred_labels.append(pred_label)

        return (
            np.array(calibrated_probs).squeeze(),
            np.array(calibrated_pred_labels).squeeze(),
        )

    def calibrate_with_prior(
        self,
        probs: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
        pred_labels: npt.NDArray[np.int64],
        pred_probs: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        return self.calibrate(probs, pred_labels)
