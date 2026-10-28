"""Use the prefilll KV cache to devode one additional token"""

import torch

from mini_vllm.generation import decode_one_token, prefill, select_greedy_token
from mini_vllm.model_loader import load_model_and_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PROMPT = "The capital of France is"
EXPECTED_FIRST_TOKEN_ID = 12095
EXPECTED_SECOND_TOKEN_ID = 13


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME,
        MODEL_REVISION,
    )

    device = model.device
    dtype = model.dtype

    encoded = tokenizer(PROMPT, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    prompt_attention_mask = encoded["attention_mask"].to(device)

    prefill_output = prefill(
        model=model,
        input_ids=prompt_ids,
        attention_mask=prompt_attention_mask,
    )

    first_token_ids = select_greedy_token(prefill_output.logits)
    kv_cache = prefill_output.past_key_values

    cache_layer_count = len(kv_cache.layers)
    first_layer_key_shape_before_decode = tuple(kv_cache.layers[0].keys.shape)
    cached_tokens_before_decode = first_layer_key_shape_before_decode[-2]

    decode_attention_mask = torch.cat(
        [
            prompt_attention_mask,
            prompt_attention_mask.new_ones(
                (prompt_attention_mask.shape[0], 1)
            ),
        ],
        dim=-1,
    )

    cache_position = torch.tensor(
        [prompt_ids.shape[1]],
        dtype=torch.long,
        device=device,
    )

    decode_output = decode_one_token(
        model=model,
        input_ids=first_token_ids,
        attention_mask=decode_attention_mask,
        kv_cache=kv_cache,
        cache_position=cache_position,
    )

    second_token_ids = select_greedy_token(decode_output.logits)

    first_token_id = first_token_ids.item()
    second_token_id = second_token_ids.item()

    first_layer_key_shape_after_decode = tuple(
        decode_output.past_key_values.layers[0].keys.shape
    )
    cached_tokens_after_decode = first_layer_key_shape_after_decode[-2]

    print(f"{device = }")
    print(f"{dtype = }")
    print(f"prefill input shape: {prompt_ids.shape}")
    print(f"kv cache layers {cache_layer_count}")
    print(f"cached tokens before decode {cached_tokens_before_decode}")
    print(f"first selected token {tokenizer.decode(first_token_id)}")
    print(f"decode input shape {first_token_ids.shape}")
    print(f"decode attention-mask shape {decode_attention_mask.shape}")
    print(f"cached position {cache_position.tolist()}")
    print(f"decode logits shape {decode_output.logits.shape}")
    print(
        "first layer key shape after decode: "
        f"{first_layer_key_shape_after_decode}"
    )
    print(f"cached tokens after decode {cached_tokens_after_decode}")
    print(f"second selected token {tokenizer.decode(second_token_id)}")

    expected_token_ids = [EXPECTED_FIRST_TOKEN_ID, EXPECTED_SECOND_TOKEN_ID]
    actual_token_ids = [first_token_id, second_token_id]

    if actual_token_ids != expected_token_ids:
        raise RuntimeError(
            f"expected reference tokens {expected_token_ids} "
            f"but got {actual_token_ids}"
        )

    print("MMatches the first two HF reference tokens: yes")


if __name__ == "__main__":
    main()
