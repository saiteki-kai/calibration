from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self, override

import numpy as np


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PredictionOutput(ABC):
    label_probs: "NDArray[float64]"
    pred_labels: "NDArray[int64]"

    def to_dict(self) -> dict[str, "NDArray[float64 | int64]"]:
        return {
            "label_probs": self.label_probs,
            "pred_labels": self.pred_labels,
        }

    def to_npz(self, path: Path) -> None:
        np.savez(path, **self.to_dict())

    @classmethod
    def from_npz(cls, path: Path) -> Self:
        return cls(**np.load(path))


@dataclass(frozen=True)
class ClassifierOutput(PredictionOutput):
    label_logits: "NDArray[float64]"

    @override
    def to_dict(self) -> dict[str, "NDArray[float64 | int64]"]:
        output = super().to_dict()
        output["label_logits"] = self.label_logits
        return output


@dataclass(frozen=True)
class CalibratorOutput(PredictionOutput):
    @override
    def to_dict(self) -> dict[str, "NDArray[float64 | int64]"]:
        return super().to_dict()
