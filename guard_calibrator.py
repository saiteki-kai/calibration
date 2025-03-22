from typing import Any, Dict, cast
from functools import partial
import numpy as np
import torch
from transformers.generation import GenerateDecoderOnlyOutput
from transformers import GenerationConfig

from datasets import Dataset

from calibration.utils import calibrate_py
from chat_template import load_chat_template
from utils import load_model, prepare_input


class GuardModel:
    def __init__(self, model_name: str, taxonomy: str, descriptions: bool = True):
        self.model, self.tokenizer = load_model(model_name)
        self.tokenizer.chat_template = load_chat_template(taxonomy, descriptions)

        self.model = torch.compile(self.model, fullgraph=True)
        self.model.eval()

    @torch.inference_mode()
    def _generate(
        self,
        example: dict[str, Any],
        max_length: int = 1024,
        max_new_tokens: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Prepare inputs
        inputs = self.tokenizer(example["text"], truncation=True, max_length=max_length, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # Configure generation parameters
        config = self._prepare_generation_config(max_new_tokens)

        # Generate outputs
        outputs = self.model.generate(**inputs, generation_config=config)  # type: ignore
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        if outputs.logits is None:
            raise ValueError("GenerationConfig.output_logits must be True to use this function.")

        return outputs.logits, outputs.sequences, inputs.input_ids.shape[-1]

    def _prepare_generation_config(self, max_new_tokens: int) -> GenerationConfig:
        """Prepare generation configuration with standard settings."""
        config = self.model.generation_config

        if config is None:
            config = GenerationConfig()

        if config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.eos_token_id

        # Set standard generation parameters
        config.max_new_tokens = max_new_tokens
        config.return_dict_in_generate = True
        config.output_logits = True
        config.do_sample = False
        config.temperature = None
        config.top_p = None
        config.top_k = None

        return config

    def _get_label_predictions(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, int],
        labels: list[str] = ["safe", "unsafe"],
        pos_label: str = "unsafe",
        label_pos: int = 1,
    ) -> tuple[float, int, torch.Tensor]:
        logits, sequences, prompt_len = outputs
        label_token_ids = [self.tokenizer.encode(lbl, add_special_tokens=False)[0] for lbl in labels]
        label_text = self.tokenizer.decode(sequences[0][prompt_len + label_pos])
        label_text = label_text.strip().lower()

        if label_text not in labels:
            raise ValueError(f"Label {label_text} not in {labels}")

        pred_label = int(label_text == pos_label)

        label_logits = tuple(logits)[label_pos][0][label_token_ids]
        label_probs = torch.softmax(label_logits, dim=-1)

        prob = float(label_probs[pred_label])
        prob = 1 - prob if pred_label == 0 else prob

        return prob, pred_label, label_probs

    def predict(self, data: Dict[str, Any], max_length: int = 2048, max_new_tokens: int = 10) -> Dict[str, Any]:
        """Get uncalibrated prediction for a single example"""
        prompt = prepare_input(data, self.tokenizer)

        # Generate logits
        outputs = self._generate(prompt, max_length, max_new_tokens)

        # Get label predictions
        prob, pred_label, pred_probs = self._get_label_predictions(outputs)

        return {"unsafe_prob": prob, "pred_label": pred_label, "pred_probs": pred_probs}

    def predict_batch(
        self,
        dataset: Dataset,
        max_length: int = 2048,
        max_new_tokens: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get uncalibrated predictions for a batch of examples"""
        dataset = dataset.map(partial(self.predict, max_length=max_length, max_new_tokens=max_new_tokens))
        dataset = cast(Dataset, dataset)

        probs = np.array(dataset["unsafe_prob"])
        labels = np.array(dataset["gt_label"])
        pred_labels = np.array(dataset["pred_label"])
        pred_probs = np.array(dataset["pred_probs"])

        return probs, labels, pred_labels, pred_probs


class BaseCalibrator:
    def __init__(self, guard_model: GuardModel):
        self.guard_model = guard_model

    def calibrate(
        self, probs: np.ndarray, labels: np.ndarray, pred_labels: np.ndarray, pred_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """To be implemented by subclasses"""
        raise NotImplementedError


class ContextFreeCalibrator(BaseCalibrator):
    def calibrate(
        self, probs: np.ndarray, labels: np.ndarray, pred_labels: np.ndarray, pred_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get context-free probability
        data_cf = {"prompt": "N/A", "response": "N/A", "gt_label": 0, "is_safe": True}
        prob_cf = self.guard_model.predict(data_cf)
        prob_cf = np.array([prob_cf, 1 - prob_cf])

        # Calibrate predictions
        calibrated_probs = []
        calibrated_pred_labels = []

        for prob in probs:
            prob = np.array([prob, 1 - prob])
            cal_prob = calibrate_py(prob, prob_cf, mode="diagonal")
            calibrated_probs.append(cal_prob[0][0])
            pred_lbl = int(np.argmin(cal_prob.reshape(-1)))
            calibrated_pred_labels.append(pred_lbl)

        return np.array(calibrated_probs), labels, np.array(calibrated_pred_labels), pred_probs


class BatchCalibrator(BaseCalibrator):
    def calibrate(
        self, probs: np.ndarray, labels: np.ndarray, pred_labels: np.ndarray, pred_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Convert to 2D probabilities and compute batch probability
        all_probs = np.stack([probs, 1 - probs], axis=1)
        batch_prob = np.mean(all_probs, axis=0)

        # Calibrate predictions
        calibrated_probs = []
        calibrated_pred_labels = []

        for prob in all_probs:
            cal_prob = calibrate_py(prob, batch_prob, mode="diagonal")
            calibrated_probs.append(cal_prob[0][0])
            pred_lbl = int(np.argmin(cal_prob.reshape(-1)))
            calibrated_pred_labels.append(pred_lbl)

        return np.array(calibrated_probs), labels, np.array(calibrated_pred_labels), pred_probs


class OriginalCalibrator(BaseCalibrator):
    def calibrate(
        self, probs: np.ndarray, labels: np.ndarray, pred_labels: np.ndarray, pred_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # No calibration, just return original predictions
        return probs, labels, pred_labels, pred_probs


class GuardModelCalibrator:
    def __init__(self, guard_model: GuardModel, eval_dataset: Dataset, test_dataset: Dataset, method: str):
        self.guard_model = guard_model
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

        # Initialize appropriate calibrator
        calibrators = {"context-free": ContextFreeCalibrator, "batch": BatchCalibrator, "original": OriginalCalibrator}

        if method not in calibrators:
            raise ValueError(f"Unknown calibration method: {method}")

        self.calibrator = calibrators[method](guard_model)

        # Get uncalibrated predictions
        self.probs, self.labels, self.pred_labels, self.pred_probs = self.guard_model.predict_batch(self.test_dataset)

        # Apply calibration
        self.calibrated_probs, self.calibrated_labels, self.calibrated_pred_labels, self.calibrated_pred_probs = (
            self.calibrator.calibrate(self.probs, self.labels, self.pred_labels, self.pred_probs)
        )

    @classmethod
    def from_predictions(
        cls, guard_model: GuardModel, method: str, probs: np.ndarray, labels: np.ndarray, pred_labels: np.ndarray
    ) -> "GuardModelCalibrator":
        """Create a calibrator using pre-computed predictions"""
        calibrators = {"context-free": ContextFreeCalibrator, "batch": BatchCalibrator, "original": OriginalCalibrator}

        if method not in calibrators:
            raise ValueError(f"Unknown calibration method: {method}")

        instance = cls.__new__(cls)
        instance.guard_model = guard_model
        instance.calibrator = calibrators[method](guard_model)

        # Store original predictions
        instance.probs = probs
        instance.labels = labels
        instance.pred_labels = pred_labels

        # Apply calibration
        instance.calibrated_probs, instance.calibrated_labels, instance.calibrated_pred_labels = (
            instance.calibrator.calibrate(probs, labels, pred_labels)
        )

        return instance

    def get_results(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return both original and calibrated results"""
        return (
            self.probs,
            self.labels,
            self.pred_labels,
            self.calibrated_probs,
            self.calibrated_labels,
            self.calibrated_pred_labels,
        )


if __name__ == "__main__":
    from datasets import DatasetDict, load_dataset
    from transformers import set_seed

    set_seed(42)

    dataset = load_dataset("PKU-Alignment/BeaverTails")
    dataset = cast(DatasetDict, dataset)

    # Take 10% of 330k_train as eval (calibration set) and 330k_test as test
    train_dataset = dataset["30k_train"].train_test_split(test_size=0.1, seed=42)

    eval_dataset = train_dataset["test"]
    test_dataset = dataset["30k_test"]

    # Initialize model and compute predictions once
    guard_model = GuardModel("meta-llama/Llama-Guard-3-1B", "llama-guard-3")
    probs, labels, pred_labels = guard_model.predict_batch(test_dataset)

    # Try different calibration methods using the same predictions
    methods = ["context-free", "batch", "original"]
    for method in methods:
        calibrator = GuardModelCalibrator.from_predictions(
            guard_model=guard_model, method=method, probs=probs, labels=labels, pred_labels=pred_labels
        )

        # Get results
        _, _, _, cal_probs, cal_labels, cal_pred_labels = calibrator.get_results()

        # Print metrics
        from calibration.utils import get_calibration_error

        print(f"\nMethod: {method}")
        print(f"Original ECE: {get_calibration_error(probs, labels, debias=False)}")
        print(f"Calibrated ECE: {get_calibration_error(cal_probs, cal_labels, debias=False)}")
