from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PredictionOutput:
    label_probs: "NDArray[float64]"
    pred_labels: "NDArray[int64]"

    def to_dict(self) -> dict[str, "NDArray[float64 | int64]"]:
        return {
            "label_probs": self.label_probs,
            "pred_labels": self.pred_labels,
        }


@dataclass(frozen=True)
class ClassifierOutput(PredictionOutput):
    label_logits: "NDArray[float64]"

    def to_dict(self) -> dict[str, "NDArray[float64 | int64]"]:
        output = super().to_dict()
        output["label_logits"] = self.label_logits
        return output


@dataclass(frozen=True)
class CalibratorOutput(PredictionOutput):
    def to_dict(self) -> dict[str, "NDArray[float64 | int64]"]:
        return super().to_dict()
