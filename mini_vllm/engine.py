"""Connect scheduled requests to batched model execution"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn

from mini_vllm.generation import (
    build_position_ids,
    decode_one_token,
    prefill,
    select_greedy_token,
)
from mini_vllm.request import Request, RequestStatus
from mini_vllm.scheduler import Scheduler


@dataclass
class ActiveBatch:
    """State shared by requests decoded together as one fixed batch"""

    requests: list[Request]
    attention_mask: Tensor
    next_token_ids: Tensor
    kv_cache: Any
    cache_position: Tensor


@dataclass
class EngineStep:
    """Describe the scheduling and cache decisions for one model call"""

    request_ids: list[str]
    admitted_request_ids: list[str]
    finished_request_ids: list[str]
    cache_action: str


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
        next_token_ids=next_token_ids,
        kv_cache=model_output.past_key_values,
        cache_position=torch.tensor(
            [input_ids.shape[1]],
            dtype=torch.long,
            device=device,
        ),
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
        input_ids=batch.next_token_ids,
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
        request.append_token(token_id)

    batch.attention_mask = attention_mask
    batch.next_token_ids = next_token_ids
    batch.kv_cache = model_output.past_key_values
    batch.cache_position = batch.cache_position + 1
    return next_token_ids


class ContinuousBatchEngine:
    """Run scheduled requests and replace them as they finish."""

    def __init__(
        self,
        model: nn.Module,
        scheduler: Scheduler,
        pad_token_id: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.pad_token_id = pad_token_id
        self.device = device
        self.active_batch: Optional[ActiveBatch] = None
        self.cache_rebuilds = 0
        self.recomputed_tokens = 0

    @property
    def has_pending_requests(self) -> bool:
        """Return whether waiting or running requests remain"""
        return bool(
            self.scheduler.waiting_requests
            or self.scheduler.running_requests
        )

    def step(self) -> EngineStep:
        """Run one scheduled model call"""
        if not self.has_pending_requests:
            raise RuntimeError("The engine has no pending requests")

        previously_running = list(self.scheduler.running_requests)
        running_requests = self.scheduler.schedule()
        admitted_requests = [
            request
            for request in running_requests
            if not any(
                request is previous_request
                for previous_request in previously_running
            )
        ]

        previous_batch_requests = (
            []
            if self.active_batch is None
            else self.active_batch.requests
        )
        same_membership = (
            len(previous_batch_requests) == len(running_requests)
            and all(
                old_request is new_request
                for old_request, new_request in zip(
                    previous_batch_requests,
                    running_requests,
                )
            )
        )

        if same_membership:
            if self.active_batch is None:
                raise RuntimeError("The active batch state is missing")
            decode_batch(model=self.model, batch=self.active_batch)
            cache_action = "decode"
        else:
            surviving_requests = [
                request
                for request in running_requests
                if any(
                    request is old_request
                    for old_request in previous_batch_requests
                )
            ]
            if surviving_requests:
                self.cache_rebuilds += 1
                self.recomputed_tokens += sum(
                    len(request.token_ids) - 1
                    for request in surviving_requests
                )
                cache_action = "rebuild"
            else:
                cache_action = "prefill"

            self.active_batch = prefill_batch(
                model=self.model,
                requests=running_requests,
                pad_token_id=self.pad_token_id,
                device=self.device,
            )

        finished_requests = self.scheduler.remove_finished_requests()
        return EngineStep(
            request_ids=[
                request.request_id for request in running_requests
            ],
            admitted_request_ids=[
                request.request_id for request in admitted_requests
            ],
            finished_request_ids=[
                request.request_id for request in finished_requests
            ],
            cache_action=cache_action,
        )
    