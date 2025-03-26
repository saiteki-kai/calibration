import warnings

from typing import Any

import numpy as np
import numpy.typing as npt

from src.guard_calibrator.calibrators.base import BaseCalibrator
from src.guard_calibrator.models.guard_model import GuardModel


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

    def compute_prior(self, probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        if probs is not None:
            msg = "Pre-computed probabilities are not used for context-free calibration"
            warnings.warn(msg)

        data = [{"prompt": self.empty_token, "response": self.empty_token}]
        _, pred_probs = self.guard_model.predict(data, **self.model_kwargs)

        return pred_probs
