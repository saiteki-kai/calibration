from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from src.core.calibrators.calibration import calibrate_py
from src.core.classifiers.guard_model import GuardModel


class BaseCalibrator(ABC):
    def __init__(self, guard_model: GuardModel) -> None:
        self.guard_model = guard_model
        self.calibration_mode = "diagonal"

    @abstractmethod
    def compute_prior(self, probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        msg = "Subclasses must implement compute_prior method"
        raise NotImplementedError(msg)

    def calibrate(
        self,
        pred_probs: npt.NDArray[np.float64],
        _pred_labels: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        prior = self.compute_prior(pred_probs)

        calibrated_probs = []
        calibrated_pred_labels = []

        for prob in pred_probs:
            cal_prob = calibrate_py(prob, prior, mode=self.calibration_mode)
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
        _labels: npt.NDArray[np.int64],
        _pred_labels: npt.NDArray[np.int64],
        _pred_probs: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        return self.calibrate(probs, _pred_labels)
