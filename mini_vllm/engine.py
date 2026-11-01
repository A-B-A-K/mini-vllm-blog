"""Connect scheduled requests to batched model execution"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from mini_vllm.generation import (
    build_position_ids,
    decode_one_token,
    prefill,
    select_greedy_token,
)
from mini_vllm.request import Request, RequestStatus


@dataclass
class ActiveBatch:
    """State shared by requests decoded together as one fixed batch"""

    requests: list[Request]
    attention_mask: Tensor
    next_token__ids: Tensor
    kv_cache: Any
    cache_position:Tensor


def build_batch_inputs(
    requests: list[Request],
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Left-pad running requests into input IDs and an attention mask"""
    if not requests:
        raise ValueError("At least one request is required")
    if any(
        request.status is not RequestStatus.RUNNING
        for request in requests
    ):
        raise RuntimeError("Every request in the batch must be running")

    padded_width = max(len(request.token_ids) for request in requests)
    input_rows = []
    attention_rows = []

    for request in requests:
        padding_length = padded_width - len(request.token_ids)
        input_rows.append(
            [pad_token_id] * padding_length + request.token_ids
        )
        attention_rows.append(
            [0] * padding_length + [1] * len(request.token_ids)
        )

    input_ids = torch.tensor(
        input_rows,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        attention_rows, 
        dtype=torch.long,
        device=device,
    )

    return input_ids, attention_mask


@torch.inference_mode()
def prefill_batch(
    model: nn.Module,
    requests: list[Request],
    pad_token_id: int,
    device: torch.device,
) -> ActiveBatch:
    """Prefill schedule requests together and retain their shared cache"""
    input_ids, attention_mask = build_batch_inputs(
        requests=requests,
        pad_token_id=pad_token_id,
        device=device,
    )
    model_output = prefill(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=build_position_ids(attention_mask),
    )
    next_token_ids = select_greedy_token(model_output.logits)

    for request, token_id in zip(
        requests,
        next_token_ids.squeeze(-1).tolist(),
    ):
        request.append_token(token_id)

    return ActiveBatch(
        requests=list(requests),
        attention_mask=attention_mask,
        next_token__ids=next_token_ids,
        kv_cache=model_output.past_key_values,
        cache_position=torch.tensor(
            [input_ids.shape[1]],
            dtype=torch.long,
            device=device,
        )
    )


@torch.inference_mode()
def decode_batch(model: nn.Module, batch: ActiveBatch) -> Tensor:
    """Advance every request by one token using the shared KV cache"""
    if any(
        request.status is not RequestStatus.RUNNING
        for request in batch.requests
    ):
        raise RuntimeError(
            "Every request must remain running during fixed-batch decoding"
        )

    attention_mask = torch.cat(
        [
            batch.attention_mask,
            batch.attention_mask.new_ones(
                (batch.attention_mask.shape[0], 1)
            ),
        ],
        dim=-1,
    )
    model_output = decode_one_token(
        model=model,
        input_ids=batch.next_token__ids,
        attention_mask=attention_mask,
        kv_cache=batch.kv_cache,
        cache_position=batch.cache_position,
        position_ids=build_position_ids(attention_mask)[:, -1:],
    )
    next_token_ids = select_greedy_token(model_output.logits)

    for request, token_id in zip(
        batch.requests,
        next_token_ids.squeeze(-1).tolist(),
    ):
        request.append(token_id)

    batch.attention_mask = attention_mask
    batch.next_token__ids = next_token_ids
    batch.kv_cache = model_output.past_key_values
    batch.cache_position = batch.cache_position + 1
    return next_token_ids