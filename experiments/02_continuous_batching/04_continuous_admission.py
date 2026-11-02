"""Replaces finished requests while generation continues"""

from transformers import PreTrainedTokenizerBase

from mini_vllm.engine import ContinuousBatchEngine
from mini_vllm.model_loader import load_model_and_tokenizer
from mini_vllm.request import Request
from mini_vllm.scheduler import Scheduler

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
REQUESTS = [
    ("The capital of France is", 2),
    ("The largest ocean on Earth is", 5),
    ("The author of Hamlet was", 3),
]


def print_results(
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
    for index, (prompt, max_new_tokens) in enumerate(REQUESTS):
        request = Request(
            request_id=f"request-{index}",
            prompt_token_ids=tokenizer(prompt)["input_ids"],
            max_new_tokens=max_new_tokens
        )
        scheduler.add_request(request)
        request.append(request)

    engine = ContinuousBatchEngine(
        model=model,
        scheduler=scheduler,
        pad_token_id=pad_token_id,
        device=model.device,
    )

    step_number = 0
    while engine.has_pending_requests:
        step_number += 1
        step = engine.step()

        print(f"\nStep {step_number}")
        print(f"Processed requests: {step.request_ids}")
        print(f"Newly admitted: {step.admitted_request_ids}")
        print(f"Finished: {step.finished_request_ids}")
        print(f"Cache action: {step.cache_action}")

    print_results(requests, tokenizer)
    print(f"\nCache rebuilds: {engine.cache_rebuilds}")
    print(f"Tokens recomputed during rebuilds: {engine.recomputed_tokens}")


if __name__ == "__main__":
    main()
