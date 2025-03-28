import logging

from typing import TYPE_CHECKING, Any

from src.core.calibrators.base import BaseCalibrator
from src.core.classifiers.guard_model import GuardModel


if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class ContextFreeCalibrator(BaseCalibrator):
    _empty_token: str

    def __init__(
        self,
        guard_model: GuardModel,
        empty_token: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(guard_model, model_kwargs)
        self._calibration_mode = "diagonal"
        self._empty_token = empty_token or " "

    def _compute_prior(self) -> "NDArray[float64]":
        data = [{"prompt": self._empty_token, "response": self._empty_token}]

        _, pred_probs = self._guard_model.predict(data, **self._model_kwargs)
        logger.info("Prior probability: %s", pred_probs)

        return pred_probs
