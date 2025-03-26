import numpy as np
import numpy.typing as npt

from src.core.calibrators.base import BaseCalibrator
from src.core.classifiers.guard_model import GuardModel


class BatchCalibrator(BaseCalibrator):
    def __init__(self, guard_model: GuardModel) -> None:
        super().__init__(guard_model)
        self.calibration_mode = "identity"

    def compute_prior(self, probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        if probs is None:
            msg = "Batch calibration requires pre-computed probabilities"
            raise ValueError(msg)

        return np.mean(probs, axis=0)
