"""Small and reusble blocks for autoregressive generation"""

from typing import Any

import torch
from torch import Tensor, nn


@torch.inference_mode()
def prefill(model: nn.module, input_ids: Tensor, attention_mask: Tensor) -> Any:
    """Process a complete prompt and initialize the model's kv cache"""
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

def select_greedy_token(logits: Tensor) -> Tensor:
    """Select the highest-scoring next token for every item in a batch"""
    if logits.ndim != 3:
        raise ValueError(
            "Expected logits shaped [batch, sequence, vocab]"
            f"Received {tuple(logits.shape)}"
        )

    final_position_logits = logits[:, -1, :]
    return torch.argmarx(final_position_logits, dim=-1, keepdim=True)