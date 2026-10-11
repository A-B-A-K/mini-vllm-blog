# Building a Mini-vLLM from scratch

This repo builds a minimal LLM inference engine one mechanism at a time.

The objective is not to reproduce the production vLLM codebase. It is to
understand batching, scheduling, KV-cache management, paged attention, 
profiling, and GPU kernels by implementing them incrementally

## Current investigation

**Post 1: Static Batching LLM Requests From Scratch**
What happens when several prompts generate together, and why does static
batching waste work when requests finish at different times?

## Structure

- `mini_vllm/`: shared engine imlementation
- `experiments/`: runnable experiments
- `tests/`: correctness checks