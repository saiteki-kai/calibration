import logging

from typing import TYPE_CHECKING

import numpy as np

from src.core.calibrators.base import BaseCalibrator
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class BatchCalibrator(BaseCalibrator):
    def __init__(self, guard_model: GuardModel) -> None:
        super().__init__(guard_model)
        self.calibration_mode = "identity"

    def compute_prior(self, probs: "NDArray[float64] | None" = None) -> "NDArray[float64]":
        if probs is None:
            msg = "Batch calibration requires pre-computed probabilities"
            raise ValueError(msg)

        avg_probs = np.mean(probs, axis=0)
        logger.info("Prior probability: %s", avg_probs)

        return avg_probs
