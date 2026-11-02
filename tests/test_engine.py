from types import SimpleNamespace

import pytest
import torch

from mini_vllm.engine import (
    ContinuousBatchEngine,
    decode_batch,
    prefill_batch,
)
from mini_vllm.request import Request
from mini_vllm.scheduler import Scheduler


class ScriptedBatchModel(torch.nn.Module):
    def __init__(self, tokens_by_call: list[list[int]]) -> None:
        super().__init__()
        self.tokens_by_call = tokens_by_call
        self.calls: list[dict[str, object]] = []

    def forward(self, **model_inputs: object) -> SimpleNamespace:
        input_ids = model_inputs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a tensor.")

        next_token_ids = self.tokens_by_call[len(self.calls)]
        self.calls.append(model_inputs)

        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            max(max(tokens) for tokens in self.tokens_by_call) + 1,
        )
        for row, token_id in enumerate(next_token_ids):
            logits[row, -1, token_id] = 1

        return SimpleNamespace(
            logits=logits,
            past_key_values=f"cache-{len(self.calls)}",
        )


def make_running_request(
    request_id: str,
    prompt_token_ids: list[int],
    max_new_tokens: int = 3,
) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
    )
    request.start()
    return request


def test_batch_prefill_then_decode_reuses_the_kv_cache() -> None:
    model = ScriptedBatchModel(
        tokens_by_call=[
            [30, 40],
            [31, 41],
        ]
    )
    requests = [
        make_running_request("request-0", [10, 11]),
        make_running_request("request-1", [20, 21, 22]),
    ]

    batch = prefill_batch(
        model=model,
        requests=requests,
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    first_call = model.calls[0]
    assert first_call["input_ids"].tolist() == [
        [0, 10, 11],
        [20, 21, 22],
    ]
    assert first_call["attention_mask"].tolist() == [
        [0, 1, 1],
        [1, 1, 1],
    ]
    assert "past_key_values" not in first_call
    assert [request.generated_token_ids for request in requests] == [
        [30],
        [40],
    ]
    assert batch.kv_cache == "cache-1"
    assert batch.cache_position.tolist() == [3]

    next_token_ids = decode_batch(model=model, batch=batch)

    second_call = model.calls[1]
    assert second_call["input_ids"].tolist() == [[30], [40]]
    assert second_call["past_key_values"] == "cache-1"
    assert second_call["attention_mask"].tolist() == [
        [0, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    assert second_call["position_ids"].tolist() == [[2], [3]]
    assert second_call["cache_position"].tolist() == [3]
    assert next_token_ids.tolist() == [[31], [41]]
    assert [request.generated_token_ids for request in requests] == [
        [30, 31],
        [40, 41],
    ]
    assert batch.kv_cache == "cache-2"
    assert batch.cache_position.tolist() == [4]


def test_batch_prefill_requires_at_least_one_request() -> None:
    model = ScriptedBatchModel(tokens_by_call=[[1]])

    with pytest.raises(ValueError, match="At least one request"):
        prefill_batch(
            model=model,
            requests=[],
            pad_token_id=0,
            device=torch.device("cpu"),
        )


def test_fixed_batch_cannot_decode_a_finished_request() -> None:
    model = ScriptedBatchModel(
        tokens_by_call=[
            [20],
            [21],
        ]
    )
    request = make_running_request(
        "request-0",
        [10],
        max_new_tokens=1,
    )
    batch = prefill_batch(
        model=model,
        requests=[request],
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="must remain running"):
        decode_batch(model=model, batch=batch)


def test_continuous_engine_replaces_finished_requests() -> None:
    model = ScriptedBatchModel(
        tokens_by_call=[
            [10, 20],
            [21, 30],
            [22, 31],
        ]
    )
    scheduler = Scheduler(max_batch_size=2)
    requests = [
        Request("request-0", [1], max_new_tokens=1),
        Request("request-1", [2], max_new_tokens=3),
        Request("request-2", [3], max_new_tokens=2),
    ]
    for request in requests:
        scheduler.add_request(request)

    engine = ContinuousBatchEngine(
        model=model,
        scheduler=scheduler,
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    first_step = engine.step()
    second_step = engine.step()
    third_step = engine.step()

    assert first_step.cache_action == "prefill"
    assert first_step.admitted_request_ids == [
        "request-0",
        "request-1",
    ]
    assert first_step.finished_request_ids == ["request-0"]

    assert second_step.cache_action == "rebuild"
    assert second_step.request_ids == ["request-1", "request-2"]
    assert second_step.admitted_request_ids == ["request-2"]

    assert third_step.cache_action == "decode"
    assert third_step.finished_request_ids == [
        "request-1",
        "request-2",
    ]

    assert model.calls[1]["input_ids"].tolist() == [
        [2, 20],
        [0, 3],
    ]
    assert model.calls[2]["input_ids"].tolist() == [[21], [30]]
    assert model.calls[2]["past_key_values"] == "cache-2"

    assert requests[0].generated_token_ids == [10]
    assert requests[1].generated_token_ids == [20, 21, 22]
    assert requests[2].generated_token_ids == [30, 31]
    assert engine.cache_rebuilds == 1
    assert engine.recomputed_tokens == 1
    assert not engine.has_pending_requests


def test_continuous_engine_rejects_a_step_without_requests() -> None:
    engine = ContinuousBatchEngine(
        model=ScriptedBatchModel(tokens_by_call=[[1]]),
        scheduler=Scheduler(max_batch_size=1),
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="no pending requests"):
        engine.step()