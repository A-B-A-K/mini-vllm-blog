import pytest
import torch

from mini_vllm.generation import decode_one_token, select_greedy_token


class RecordingModel(torch.nn.Module):
    def forward(self, **model_inputs: object) -> str:
        self.model_inputs = model_inputs
        return "model output"


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