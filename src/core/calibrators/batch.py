import logging

from typing import TYPE_CHECKING, Any, override

import numpy as np

from src.core.calibrators.base import BaseCalibrator
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class BatchCalibrator(BaseCalibrator):
    _probs: "NDArray[float64]"

    def __init__(
        self,
        guard_model: GuardModel,
        probs: "NDArray[float64]",
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(guard_model, model_kwargs)
        self._probs = probs

    @override
    def _calibrate_prob(self, prob: "NDArray[float64]", prior: "NDArray[float64]") -> "NDArray[float64]":
        num_classes = prob.shape[0]

        W = np.identity(num_classes)
        b = -1 * np.log(prior)
        cal_prob = np.matmul(W, np.log(prob + 10e-6)) + b
        cal_prob = np.exp(cal_prob)

        return cal_prob / np.sum(cal_prob)

    def _compute_prior(self) -> "NDArray[float64]":
        avg_probs = np.mean(self._probs, axis=0)
        logger.info("Prior probability: %s", avg_probs)

        return avg_probs
