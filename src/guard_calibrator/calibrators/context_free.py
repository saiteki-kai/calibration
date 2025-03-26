import numpy as np
import numpy.typing as npt
from typing import Any
from .base import BaseCalibrator
from ..models.guard_model import GuardModel


class ContextFreeCalibrator(BaseCalibrator):
    def __init__(self, guard_model: GuardModel, model_kwargs: dict[str, Any]):
        super().__init__(guard_model)
        self.calibration_mode = "diagonal"
        self.model_kwargs = model_kwargs

    def compute_prior(self, probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        data_cf = {"prompt": " ", "response": " "}
        _, pred_probs = self.guard_model.predict([data_cf], **self.model_kwargs)

        return pred_probs
