from functools import partial
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset, load_dataset
from torch import bfloat16, dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def load_model(
    model_name: str,
    tokenizer_name: str | None = None,
    torch_dtype: dtype = bfloat16,
    trust_remote_code: bool = True,
    device_map: str = "cuda",
) -> tuple[PreTrainedModel, "Tokenizer"]:
    if tokenizer_name is None:
        tokenizer_name = model_name

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )

    return model, tokenizer


def load_beavertails(tokenizer: "Tokenizer") -> Dataset:
    dataset = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")

    dataset = dataset.map(partial(prepare_input, tokenizer=tokenizer))
    dataset = dataset.remove_columns(["prompt", "response", "is_safe", "category"])

    return cast(Dataset, dataset)


def prepare_input(example: dict[str, Any], tokenizer: "Tokenizer") -> dict[str, Any]:
    chat = [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["response"]}]
    text = str(tokenizer.apply_chat_template(chat, tokenize=False))

    return {"text": text, "gt_label": int(example["is_safe"] is False)}
