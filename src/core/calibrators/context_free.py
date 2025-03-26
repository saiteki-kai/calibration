import logging

from typing import TYPE_CHECKING, Any

from src.core.calibrators.base import BaseCalibrator
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class ContextFreeCalibrator(BaseCalibrator):
    def __init__(
        self,
        guard_model: GuardModel,
        empty_token: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(guard_model)
        self.calibration_mode = "diagonal"
        self.empty_token = empty_token or " "
        self.model_kwargs = model_kwargs or {}

    def compute_prior(self, probs: "NDArray[float64] | None" = None) -> "NDArray[float64]":
        if probs is not None:
            logger.warning("Predicted probabilities are not used for context-free calibration")

        data = [{"prompt": self.empty_token, "response": self.empty_token}]
        _, pred_probs = self.guard_model.predict(data, **self.model_kwargs)

        logger.info("Prior probability: %s", pred_probs)

        return pred_probs
