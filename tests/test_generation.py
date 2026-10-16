from types import SimpleNamespace

import pytest
import torch

from mini_vllm.generation import (
    build_position_ids,
    decode_one_token,
    generate_greedy,
    select_greedy_token,
)


class RecordingModel(torch.nn.Module):
    def forward(self, **model_inputs: object) -> str:
        self.model_inputs = model_inputs
        return "model output"


class ScriptedModel(torch.nn.Module):
    def __init__(self, next_token_ids: list[int]) -> None:
        super().__init__()
        self.next_token_ids = next_token_ids
        self.calls: list[dict[str, object]] = []

    def forward(self, **model_inputs: object) -> SimpleNamespace:
        input_ids = model_inputs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a tensor.")

        next_token_id = self.next_token_ids[len(self.calls)]
        self.calls.append(model_inputs)

        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            max(self.next_token_ids)+1,
            device=input_ids.device,
        )
        logits[:, -1, next_token_id] = 1

        return SimpleNamespace(
            logits=logits,
            past_key_values=f"cache-{len(self.calls)}",
        )


def test_select_greedy_token_uses_final_sequence_position() -> None:
    logits = torch.tensor(
        [
            [[9.0, 1.0, 0.0], [0.0, 5.0, 1.0]],
            [[0.0, 8.0, 1.0], [7.0, 1.0, 2.0]],
        ]
    )

    token_ids = select_greedy_token(logits)

    assert token_ids.tolist() == [[1], [0]]


def test_select_greedy_token_requires_three_dimensions() -> None:
    logits = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="batch, sequence, vocabulary"):
        select_greedy_token(logits)


def test_build_position_ids_ignores_left_padding() -> None:
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    position_ids = build_position_ids(attention_mask)

    assert position_ids.tolist() == [
        [1, 1, 0, 1, 2],
        [0, 1, 2, 3, 4],
    ]


def test_decode_one_token_passes_cache_and_position_to_model() -> None:
    model = RecordingModel()
    input_ids = torch.tensor([[7], [8]])
    attention_mask = torch.ones(2, 4, dtype=torch.long)
    kv_cache = object()
    cache_position = torch.tensor([3])

    output = decode_one_token(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
        cache_position=cache_position,
    )

    assert output == "model output"
    assert torch.equal(model.model_inputs["input_ids"], input_ids)
    assert torch.equal(model.model_inputs["attention_masl"], attention_mask)
    assert model.model_inputs["past_ley_values"] is kv_cache
    assert torch.equal(model.model_inputs["cache_position"], cache_position)
    assert model.model_inputs["use_cache"] is True


def test_decode_one_token_rejects_more_than_one_token() -> None:
    model = RecordingModel()
    input_ids = torch.tensor([[7, 8]])

    with pytest.raises(ValueError, match=r"\[batch, 1\]"):
        decode_one_token(
            model=model,
            input_ids=input_ids,
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            kv_cache=object(),
            cache_position=torch.tensor([2, 3]),
        )


def test_generate_greedy_reuses_cache_until_requested_length() -> None:
    model = ScriptedModel(next_token_ids=[2, 1, 3])
    input_ids = torch.tensor([4, 5, 6])
    attention_mask = torch.ones(1, 3, dtype=torch.long)

    generated_ids = generate_greedy(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=3,
    )

    assert generated_ids.tolist() == [[2, 1, 3]]
    assert len(model.calls) == 3

    assert "past_key_values" not in model.calls[0]
    assert model.calls[1]["past_key_values"] == "cache-1"
    assert model.calls[2]["past_key_values"] == "cache-2"

    assert model.calls[1]["cache_position"].tolist() == [3]
    assert model.calls[2]["cache_position"].tolist() == [4]

    assert model.calls[0]["attention_mask"].shape == (1, 3)
    assert model.calls[1]["attention_mask"].shape == (1, 4)
    assert model.calls[2]["attention_mask"].shape == (1, 5)


def test_generate_greedy_requires_at_least_one_new_token() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        generate_greedy(
            model=RecordingModel(),
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            max_new_tokens=0,
        )


def test_generate_greedy_tracks_positions_in_a_left_padded_batch() -> None:
    model = ScriptedModel(next_token_ids=[2, 1])
    input_ids = torch.tensor(
        [
            [0, 4, 5],
            [6, 7, 8],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    generate_greedy(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
    )

    assert model.calls[0]["position_ids"].tolist() == [
        [1, 0, 1],
        [0, 1, 2],
    ]
    assert model.calls[1]["position_ids"].tolist() == [[2], [3]]
    assert model.calls[1]["cache_position"].tolist() == [3]