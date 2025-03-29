import logging

from typing import TYPE_CHECKING, Any

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
        self._calibration_mode = "diagonal"
        if token is None:
            token = ["N/A", " ", "[MASK]"]
        self._token = [token] if isinstance(token, str) else token

    def _compute_prior(self) -> "NDArray[float64]":
        data = [{"prompt": token, "response": token} for token in self._token]

        _, pred_probs = self._guard_model.predict(data, **self._model_kwargs)
        logger.info("Prior probabilities: %s", zip(self._token, pred_probs))

        average_probs = np.mean(pred_probs, axis=0)
        logger.info("Average probability: %s", average_probs)

        return average_probs
