"""Measure the benefits and wasted work of naive static batching"""

from collections.abc import Callable
from functools import partial
from statistics import median
from time import perf_counter

import torch
from torch import Tensor, nn

from mini_vllm.generation import generate_greedy
from mini_vllm.model_loader import load_model_and_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WARMUP_RUNS = 3
MEASURED_RUNS = 10

WORKLOADS = [
    (
        "Homogeneous requests",
        [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
            "The capital of Spain is",
        ],
        [16, 16, 16, 16],
    ),
    (
        "Heterogeneous requests",
        [
            "France is",
            "The capital of France is",
            (
                "Complete this sentence about European geography: "
                "The capital of France is"
            ),
            (
                "Write a short factual paragraph explain which city is the "
                "capital of France and why it is important to the country:"
            ),
        ],
        [4, 8, 16, 32],
    ),
]


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work so the timer measures completed execution"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_median_seconds(
        operation: Callable[[], object],
        device: torch.device,
) -> float:
    """Warm up an operation and return its median e2e runtime"""
    for _ in range(WARMUP_RUNS):
        operation()

    synchronize(device)
    measured_seconds = []

    for _ in range(MEASURED_RUNS):
        synchronize(device)
        start_time = perf_counter()
        operation()
        synchronize(device)
        measured_seconds.append(perf_counter() - start_time)

    return median(measured_seconds)


def generate_independently(
        model: nn.Module,
        request_inputs: list[tuple[Tensor, Tensor]],
        output_lengths: list[int],
) -> list[Tensor]:
    """Generate each request for only its requested output length"""
    return [
        generate_greedy(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        for (input_ids, attention_mask), max_new_tokens in zip(
            request_inputs,
            output_lengths,
        )
    ]


def generate_static_batch(
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor,
        output_lenghts: list[int],
) -> Tensor: 
    """Keep every request active until the longest request is complete"""
    return generate_greedy(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max(output_lenghts),
    )


def print_workload_results(
    name: str,
    prompt_lengths: list[int],
    output_lengths: list[int],
    sequential_seconds: float,
    batched_seconds: float,
) -> None:
    """Print timing and utilization metrics for one workload"""
    request_count = len(prompt_lengths)
    padded_prompt_width = max(prompt_lengths)
    longest_output = max(output_lengths)

    useful_prefill_slots = sum(prompt_lengths)
    processed_prefill_slots = request_count * padded_prompt_width
    wasted_prefill_slots = processed_prefill_slots - useful_prefill_slots
    prefill_utilization = useful_prefill_slots / processed_prefill_slots

    useful_decode_slots = sum(length - 1 for length in output_lengths)
    processed_decode_slots = request_count * (longest_output - 1)
    wasted_decode_slots = processed_decode_slots - useful_decode_slots
    decode_utilization = useful_decode_slots / processed_decode_slots

    useful_output_tokens = sum(output_lengths)
    sequential_requests_per_second = request_count / sequential_seconds
    batched_requests_per_second = request_count / batched_seconds
    sequential_tokens_per_second = useful_output_tokens / sequential_seconds
    batched_tokens_per_second = useful_output_tokens / batched_seconds

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(f"Prompt lengths: {prompt_lengths}")
    print(f"Requested output lengths: {output_lengths}")
    print(f"Sequential median: {sequential_seconds * 1_000:.2f} ms")
    print(f"Static batch median: {batched_seconds * 1_000:.2f} ms")
    print(f"Static batch speedup: {sequential_seconds / batched_seconds:.2f}x")
    print()
    print("Useful throughput:")
    print(f"  Sequential requests/s: {sequential_requests_per_second:.2f}")
    print(f"  Static batch requests/s: {batched_requests_per_second:.2f}")
    print(f"  Sequential output tokens/s: {sequential_tokens_per_second:.2f}")
    print(f"  Static batch output tokens/s: {batched_tokens_per_second:.2f}")
    print()
    print("Static batch slot utilization:")
    print(
        f"  Prefill: {useful_prefill_slots}/{processed_prefill_slots} useful "
        f"({prefill_utilization:.1%}), {wasted_prefill_slots} padding slots"
    )
    print(
        f"  Decode: {useful_decode_slots}/{processed_decode_slots} useful "
        f"({decode_utilization:.1%}), "
        f"{wasted_decode_slots} finished-request slots"
    )


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME,
        MODEL_REVISION,
    )
    device = model.device

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Device: {device}")
    print(f"Data type: {model.dtype}")
    print(f"Warmup runs per method: {WARMUP_RUNS}")
    print(f"Measured runs per method: {MEASURED_RUNS}")
    print("Model loading, tokenization, and device transfer are not timed.")

    for name, prompts, output_lengths in WORKLOADS:
        request_inputs = []
        for prompt in prompts:
            encoded_request = tokenizer(prompt, return_tensors="pt")
            request_inputs.append(
                (
                    encoded_request["input_ids"].to(device),
                    encoded_request["attention_mask"].to(device),
                )
            )

        encoded_batch = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )
        batch_input_ids = encoded_batch["input_ids"].to(device)
        batch_attention_mask = encoded_batch["attention_mask"].to(device)

        sequential_seconds = measure_median_seconds(
            partial(
                generate_independently,
                model=model,
                request_inputs=request_inputs,
                output_lengths=output_lengths,
            ),
            device=device,
        )
        batched_seconds = measure_median_seconds(
            partial(
                generate_static_batch,
                model=model,
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_lengths=output_lengths,
            ),
            device=device,
        )

        print_workload_results(
            name=name,
            prompt_lengths=batch_attention_mask.sum(dim=-1).tolist(),
            output_lengths=output_lengths,
            sequential_seconds=sequential_seconds,
            batched_seconds=batched_seconds,
        )


if __name__ == "__main__":
    main()