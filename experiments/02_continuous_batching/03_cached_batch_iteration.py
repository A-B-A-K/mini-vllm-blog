"""Advance a scheduled batch with prefill followed by cached decoding."""

from transformers import PreTrainedTokenizerBase

from mini_vllm.engine import decode_batch, prefill_batch
from mini_vllm.model_loader import load_model_and_tokenizer
from mini_vllm.request import Request
from mini_vllm.scheduler import Scheduler

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PROMPTS = [
    "The capital of France is",
    "The largest ocean on Earth is",
    "The author of Hamlet was",
]
MAX_NEW_TOKENS = 4


def print_requests(
        requests: list[Request],
        tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Print the generated state of each request"""
    for request in requests:
        continuation = tokenizer.decode(request.generated_token_ids)
        print(
            f"{request.request_id}"
            f"\nIDs={request.generated_token_ids}, "
            f"\ntext={continuation}"
        )


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME,
        MODEL_REVISION,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("The tokenizer does not define a padding token")


    scheduler = Scheduler(max_batch_size=2)
    requests = []
    for index, prompt in enumerate(PROMPTS):
        prompt_token_ids = tokenizer(prompt)["input_ids"]
        request = Request(
            request_id=f"request-{index}",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        scheduler.add_request(request)
        requests.append(request)

    running_requests = scheduler.schedule()
    waiting_requests_ids = [
        request.request_id for request in scheduler.waiting_requests
    ]

    print(f"Device {model.device}")
    print(
        "Running requests: "
        f"{[request.request_id for request in running_requests]}"
    )
    print(f"Waiting requests: {waiting_requests_ids}")

    batch = prefill_batch(
        model=model,
        requests=running_requests,
        pad_token_id=pad_token_id,
        device=model.device,
    )
    key_shape_after_prefill = tuple(
        batch.kv_cache.layers[0].keys.shape
    )

    print("\nAfter batched prefill")
    print_requests(running_requests, tokenizer)
    print(f"Attention-mask shape: {tuple(batch.attention_mask.shape)}")
    print(f"First-layer key shape: {key_shape_after_prefill}")

    decode_batch(model=model, batch=batch)
    key_shape_after_decode = tuple(
        batch.kv_cache.layers[0].keys.shape
    )

    print("\nAfter one cached decoding step")
    print_requests(running_requests, tokenizer)
    print(f"Next-token tensor shape: {tuple(batch.next_token_ids.shape)}")
    print(f"Attention-mask shape: {tuple(batch.attention_mask.shape)}")
    print(f"First-layer key shape: {key_shape_after_decode}")
    print(f"Next cache position: {batch.cache_position.tolist()}")


if __name__ == "__main__":
    main()
