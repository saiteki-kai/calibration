import numpy as np
import numpy.typing as npt

from .base import BaseCalibrator
from ..models.guard_model import GuardModel


class BatchCalibrator(BaseCalibrator):
    def __init__(self, guard_model: GuardModel):
        super().__init__(guard_model)
        self.calibration_mode = "identity"

    def compute_prior(self, probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        if probs is None:
            raise ValueError("Batch calibration requires pre-computed probabilities")

        return np.mean(probs, axis=0)
