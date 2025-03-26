from typing import cast

import numpy as np
import numpy.typing as npt
import torch

from tqdm import tqdm
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput

from src.guard_calibrator.utils.chat_template import load_chat_template
from src.guard_calibrator.utils.loader import load_model


class GuardModel:
    def __init__(
        self,
        model_name: str,
        taxonomy: str,
        labels: list[str] = ["safe", "unsafe"],
        pos_label: str = "unsafe",
        label_token_pos: int = 1,
        descriptions: bool = True,
    ):
        self.model, self.tokenizer = load_model(model_name)
        self.model.eval()
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

        self.tokenizer.chat_template = load_chat_template(taxonomy, descriptions)
        self.label_token_ids = [self.tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]

        self.labels = labels
        self.pos_label = pos_label
        self.label_token_pos = label_token_pos

    def prepare_input(self, example: dict[str, str]) -> str:
        chat = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]

        return str(self.tokenizer.apply_chat_template(chat, tokenize=False))

    @torch.inference_mode()
    def predict(
        self,
        data: list[dict[str, str]],
        max_new_tokens: int = 10,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        self.generation_config = self._prepare_generation_config(max_new_tokens)

        results = []
        for example in tqdm(data, desc="Computing predictions"):
            result = self._predict(example)
            results.append(result)

        pred_labels, label_probs = zip(*results)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        label_probs = np.asarray(label_probs, dtype=np.float64)

        return pred_labels, label_probs

    def _generate(self, text: str) -> tuple[tuple[torch.Tensor], torch.Tensor, int]:
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate outputs
        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        if outputs.logits is None:
            raise ValueError("GenerationConfig.output_logits must be True to use this function.")

        return outputs.logits, outputs.sequences, inputs["input_ids"].shape[-1]

    def _prepare_generation_config(self, max_new_tokens: int) -> GenerationConfig:
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
        self, outputs: tuple[tuple[torch.Tensor], torch.Tensor, int]
    ) -> tuple[int, npt.NDArray[np.float64]]:
        logits, sequences, prompt_len = outputs

        label_token_id = sequences[0][prompt_len + self.label_token_pos]
        label_text = self.tokenizer.decode(label_token_id)
        label_text = label_text.strip().lower()

        if label_text not in self.labels:
            raise ValueError(f"Predicted label '{label_text}' not in allowed labels: {self.labels}")

        pred_label = self.labels.index(label_text)

        # Get logits and probabilities for both classes
        label_logits = tuple(logits)[self.label_token_pos][0][self.label_token_ids]
        label_probs = torch.softmax(label_logits, dim=-1)

        return pred_label, label_probs.detach().cpu().numpy()

    def _predict(self, data: dict[str, str]) -> tuple[int, npt.NDArray[np.float64]]:
        prompt = self.prepare_input(data)
        outputs = self._generate(prompt)
        return self._get_label_predictions(outputs)
