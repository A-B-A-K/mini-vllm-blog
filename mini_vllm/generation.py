"""Small and reusble blocks for autoregressive generation"""

from typing import Any, Optional

import torch
from torch import Tensor, nn


def build_position_ids(attention_mask: Tensor) -> Tensor:
    """Assign positions to real tokens while ignoring left-padding tokens."""
    if attention_mask.ndim != 2:
        raise ValueError(
            "Expected attention_mask shaped [batch, sequence], "
            f"but received {tuple(attention_mask.shape)}"
        )

    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


@torch.inference_mode()
def prefill(model: nn.Module, 
            input_ids: Tensor, 
            attention_mask: Tensor,
            position_ids: Optional[Tensor] = None,
            ) -> Any:
   
    """Process a complete prompt and initialize the model's kv cache"""
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )


def select_greedy_token(logits: Tensor) -> Tensor:
    """Select the highest-scoring next token for every item in a batch"""
    if logits.ndim != 3:
        raise ValueError(
            "Expected logits shaped [batch, sequence, vocab] "
            f"Received {tuple(logits.shape)}"
        )

    final_position_logits = logits[:, -1, :]
    return torch.argmax(final_position_logits, dim=-1, keepdim=True)


@torch.inference_mode()
def decode_one_token(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    kv_cache: Any,
    cache_position: Tensor,
    position_ids: Optional[Tensor] = None
) -> Any:
    """Prcess one new token while reusing the keys and values already cached"""
    if input_ids.ndim != 2 or input_ids.shape[1] != 1:
        raise ValueError(
            "Expected one token per batch item shaped [batch, 1], "
            f"But receibed {tuple(input_ids.shape)}"
        )

    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=kv_cache,
        cache_position=cache_position,
        position_ids=position_ids,
        use_cache=True,
    )


@torch.inference_mode()
def generate_greedy(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Generate a fixed number of new tokens while reusing the KV cache"""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    prefill_position_ids = build_position_ids(attention_mask)
    model_output = prefill(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=prefill_position_ids,
    )
    next_token_ids = select_greedy_token(model_output.logits)
    generated_token_ids = [next_token_ids]

    kv_cache = model_output.past_key_values
    decode_attention_mask = attention_mask
    cache_position = torch.tensor(
        [input_ids.shape[1]],
        dtype=torch.long,
        device=input_ids.device,
    )

    for _ in range(max_new_tokens - 1):
        decode_attention_mask = torch.cat(
            [
                decode_attention_mask,
                decode_attention_mask.new_ones(
                    (decode_attention_mask.shape[0], 1)
                    ),
            ],
            dim=-1,
        )

        decode_position_ids = build_position_ids(
            decode_attention_mask
        )[:, -1:]
        
        model_output = decode_one_token(
            model=model,
            input_ids=next_token_ids,
            attention_mask=decode_attention_mask,
            kv_cache=kv_cache,
            cache_position=cache_position,
            position_ids=decode_position_ids,
        )
        next_token_ids = select_greedy_token(model_output.logits)
        generated_token_ids.append(next_token_ids)

        kv_cache = model_output.past_key_values
        cache_position = cache_position + 1

    return torch.cat(generated_token_ids, dim=-1)