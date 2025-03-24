import numpy as np
import numpy.typing as npt

from .base import BaseCalibrator


class ContextFreeCalibrator(BaseCalibrator):
    def compute_prior(self, _probs: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        data_cf = {"prompt": " ", "response": " ", "true_label": 0, "is_safe": True}
        _, pred_probs = self.guard_model.predict(data_cf)

        return pred_probs
