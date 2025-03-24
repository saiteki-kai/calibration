import numpy as np
import numpy.typing as npt
from typing import Literal

from .base import BaseCalibrator


class BatchCalibrator(BaseCalibrator):
    def compute_prior(self, probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        if probs is None:
            raise ValueError("Batch calibration requires pre-computed probabilities")

        return np.mean(probs, axis=0)

    def get_calibration_mode(self) -> Literal["diagonal", "identity"]:
        return "identity"
