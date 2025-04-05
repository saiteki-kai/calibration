from pathlib import Path
from typing import Any

from datasets import Dataset

from src.core.classifiers.guard_model import GuardModel
from src.core.types import ClassifierOutput


def compute_or_load_predictions(
    guard_model: GuardModel,
    dataset: Dataset,
    output_path: Path,
    model_kwargs: dict[str, Any],
) -> ClassifierOutput:
    pred_output_path = Path(output_path)
    pred_output_path.parent.mkdir(parents=True, exist_ok=True)

    if pred_output_path.exists():
        output = ClassifierOutput.from_npz(pred_output_path)
        print(f"Loaded predictions from {pred_output_path}")
    else:
        output = guard_model.predict(dataset.to_list(), model_kwargs=model_kwargs)
        output.to_npz(pred_output_path)
        print(f"Saved predictions to {pred_output_path}")

    return output
