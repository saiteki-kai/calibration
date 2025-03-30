from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from tqdm import tqdm
from transformers import GenerationConfig


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray
    from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
    from transformers.generation import GenerateDecoderOnlyOutput


from src.core.classifiers.chat_template import load_chat_template
from src.core.classifiers.loader import load_model


class GuardModel:
    _model: "PreTrainedModel"
    _tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast"
    _pos_label: str
    _label_token_pos: int
    _labels: list[str]
    _label_token_ids: list[int]

    def __init__(
        self,
        model_name: str,
        taxonomy: str,
        labels: list[str] = ["safe", "unsafe"],
        pos_label: str = "unsafe",
        label_token_pos: int = 1,
        descriptions: bool = True,
    ) -> None:
        self._model, self._tokenizer = load_model(model_name)
        self._model.eval()

        self._tokenizer.chat_template = load_chat_template(taxonomy, descriptions)
        self._label_token_ids = [self._tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]

        self._labels = labels
        self._pos_label = pos_label
        self._label_token_pos = label_token_pos

    @torch.inference_mode()
    def predict(
        self,
        data: list[dict[str, str]],
        max_new_tokens: int = 10,
    ) -> tuple["NDArray[int64]", "NDArray[float64]", "NDArray[float64]"]:
        self._generation_config = self._prepare_generation_config(max_new_tokens)

        results = []
        for example in tqdm(data, desc="Computing predictions"):
            result = self._predict(example)
            results.append(result)

        pred_labels, label_probs, label_logits = zip(*results)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        label_probs = np.asarray(label_probs, dtype=np.float64)
        label_logits = np.asarray(label_logits, dtype=np.float64)

        return pred_labels, label_probs, label_logits

    def _predict(self, data: dict[str, str]) -> tuple[int, "NDArray[float64]", "NDArray[float64]"]:
        prompt = self._prepare_input(data)
        outputs = self._generate(prompt)

        return self._extract_predictions(outputs)

    def _prepare_input(self, example: dict[str, str]) -> str:
        chat = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]

        return str(self._tokenizer.apply_chat_template(chat, tokenize=False))

    def _generate(self, text: str) -> tuple[tuple[torch.Tensor], torch.Tensor, int]:
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate outputs
        outputs = self._model.generate(**inputs, generation_config=self._generation_config)
        outputs = cast("GenerateDecoderOnlyOutput", outputs)

        if outputs.logits is None:
            msg = "GenerationConfig.output_logits must be True to use this function."
            raise ValueError(msg)

        return outputs.logits, outputs.sequences, inputs["input_ids"].shape[-1]

    def _extract_predictions(
        self,
        outputs: tuple[tuple[torch.Tensor], torch.Tensor, int],
    ) -> tuple[int, "NDArray[float64]", "NDArray[float64]"]:
        logits, sequences, prompt_len = outputs

        label_token_id = sequences[0][prompt_len + self._label_token_pos]
        label_text = self._tokenizer.decode(label_token_id)
        label_text = label_text.strip().lower()

        if label_text not in self._labels:
            msg = f"Predicted label '{label_text}' not in allowed labels: {self._labels}"
            raise ValueError(msg)

        pred_label = self._labels.index(label_text)

        # Get logits and probabilities for both classes
        label_logits = tuple(logits)[self._label_token_pos][0][self._label_token_ids]
        label_probs = torch.softmax(label_logits, dim=-1)

        return pred_label, label_probs.detach().cpu().numpy(), label_logits.detach().cpu().numpy()

    def _prepare_generation_config(self, max_new_tokens: int) -> GenerationConfig:
        config = self._model.generation_config

        if config is None:
            config = GenerationConfig()

        if config.pad_token_id is None:
            config.pad_token_id = self._tokenizer.eos_token_id

        # Set standard generation parameters
        config.max_new_tokens = max_new_tokens
        config.return_dict_in_generate = True
        config.output_logits = True
        config.do_sample = False
        config.temperature = None
        config.top_p = None
        config.top_k = None

        return config
