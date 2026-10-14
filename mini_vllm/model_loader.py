"""Load an HF causal language model for local inference"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_model_and_tokenizer(
    model_name: str,
    revision: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load one model and tokenizer on the best available local device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    return model, tokenizer
