# Building a Mini-vLLM from scratch

This repo builds a minimal LLM inference engine one mechanism at a time.

We use pretrained model weights and gradually implemement the code that surrounds the model during generation. The objective is not to reproduce production vLLM but to understand why mechanisms such as batching, scheduling and KV-cache management exist by building minimal versions ourselves.

## Experiment series
The project is divided into three rounds. Each introducing a larger inference-engine problem and supports a separate blog post. 

1. Static Batching: What happens when multiple requests run as one fixed batch
2. Continuous Batching: How can requests enter and leave while generation continues?
3. KV-cache management: How can the engine store and reuse cached tokens efficiently? 

Later work will build on these foundations to profile the engine and implement targeted PyTorch or Triton optimizations.

The order of experiments is part of the repo's design. The first script relies on HF's high-level generation machinery. Each following script replaces another hidden operation with an implementation written by hand: token selection, cached decoding, the generation loop, batching, and eventually scheduling and cache management. Interested readers should therefore follow the numbered experiments in order.

## Repo structure
```
mini_vllm/      Reusable inference-engine implementation
experiments/    Experiments grouped by round and blog post
tests/          Focused correctness test
```
Completed experiment scripts remain snapshots of how much of the inference engine had been implemented at that point. Reusable behavior moves into `mini_vllm/` so the next experiment can extend it without hiding the pgression from the reader.

## Current round
Round 1 builds the most naive multi-request inference engine:
1. Record HF's generation as a reference
2. Select one token without `model.generate()`
3. Decode with a KV cache
4. Build the complete greedy generation loop
5. Generate several prompts as one static batch
6. Measure the performance and wasted work of that batch

The current model is `Qwen/Qwen2.5-0.5B`, using a pinned revision and greedy decoding so the exact token IDs can be compared.

## Setup
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```
## Run
Run the project checks
```
pytest
ruff check.
```
Experiments are run form the repository root
```
python experiments/<round>/<experiment>.py
```