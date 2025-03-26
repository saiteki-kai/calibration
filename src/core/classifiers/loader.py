from typing import TYPE_CHECKING

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


if TYPE_CHECKING:
    from torch import dtype
    from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

    Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def load_model(
    model_name: str,
    tokenizer_name: str | None = None,
    torch_dtype: "dtype" = torch.bfloat16,
    trust_remote_code: bool = True,
    device_map: str = "cuda",
) -> tuple["PreTrainedModel", "Tokenizer"]:
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
