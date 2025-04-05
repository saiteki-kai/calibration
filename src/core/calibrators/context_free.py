import logging

from typing import TYPE_CHECKING, Any, override

import numpy as np

from src.core.calibrators.base import BaseCalibrator
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class ContextFreeCalibrator(BaseCalibrator):
    _token: str | list[str]

    def __init__(
        self,
        guard_model: GuardModel,
        token: str | list[str] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(guard_model, model_kwargs)

        if token is None:
            token = ["N/A", " ", "[MASK]"]

        self._token = [token] if isinstance(token, str) else token

    @override
    def _calibrate_prob(self, prob: "NDArray[float64]", prior: "NDArray[float64]") -> "NDArray[float64]":
        num_classes = prob.shape[0]

        W = np.linalg.inv(np.identity(num_classes) * prior)
        cal_prob = np.matmul(W, prob)

        return cal_prob / np.sum(cal_prob)

    def _compute_prior(self) -> "NDArray[float64]":
        data = [{"prompt": token, "response": token} for token in self._token]

        output = self._guard_model.predict(data, **self._model_kwargs)
        logger.info("Prior probabilities: %s", list(zip(self._token, output.label_probs)))

        average_probs = np.mean(output.label_probs, axis=0)
        logger.info("Average probability: %s", average_probs)

        return average_probs
