from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from tqdm import tqdm

from src.core.classifiers.guard_model import GuardModel
from src.core.types import CalibratorOutput, ClassifierOutput


if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray


class BaseCalibrator(ABC):
    _guard_model: GuardModel
    _model_kwargs: dict[str, Any]

    def __init__(self, guard_model: GuardModel, model_kwargs: dict[str, Any] | None = None) -> None:
        self._guard_model = guard_model
        self._model_kwargs = model_kwargs or {}

    def calibrate(self, preds: ClassifierOutput) -> CalibratorOutput:
        prior = self._compute_prior()

        cal_probs = []
        cal_pred_labels = []

        for prob in tqdm(preds.label_probs, desc="Calibrating predictions"):
            cal_prob = self._calibrate_prob(prob, prior)
            cal_probs.append(cal_prob)

            pred_label = int(np.argmax(cal_prob))
            cal_pred_labels.append(pred_label)

        return CalibratorOutput(
            label_probs=np.asarray(cal_probs),
            pred_labels=np.asarray(cal_pred_labels),
        )

    @abstractmethod
    def _calibrate_prob(self, prob: "NDArray[float64]", prior: "NDArray[float64]") -> "NDArray[float64]":
        msg = "Subclasses must implement calibrate_prob method"
        raise NotImplementedError(msg)

    @abstractmethod
    def _compute_prior(self) -> "NDArray[float64]":
        msg = "Subclasses must implement compute_prior method"
        raise NotImplementedError(msg)
