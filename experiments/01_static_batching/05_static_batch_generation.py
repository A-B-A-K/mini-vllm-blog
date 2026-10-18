"""Generate several differently sized prompts together in one static batch"""

from mini_vllm.generation import build_position_ids, generate_greedy
from mini_vllm.model_loader import load_model_and_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PROMPTS = [
    "The capital of France is",
    "France's capital is",
    "The city known as the capital of France is",
]
MAX_NEW_TOKENS = 4

def main() -> None:
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME,
        MODEL_REVISION,
    )
    device = model.device

    independent_token_ids = []
    for prompt in PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt")
        generated_ids = generate_greedy(
            model=model,
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        independent_token_ids.append(generated_ids[0].tolist())

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_batch = tokenizer(
        PROMPTS,
        padding=True,
        return_tensors="pt",
    )
    batch_input_ids = encoded_batch["input_ids"].to(device)
    batch_attention_mask = encoded_batch["attention_mask"].to(device)
    batch_position_ids = build_position_ids(batch_attention_mask)

    batch_generated_ids = generate_greedy(
        model=model,
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    batched_token_ids = batch_generated_ids.tolist()

    prompt_lengths = batch_attention_mask.sum(dim=-1).tolist()
    padded_width = batch_input_ids.shape[1]

    print(f"Device: {device}")
    print(f"Data type: {model.dtype}")
    print(f"Batch shape: {tuple(batch_input_ids.shape)}")
    print(f"Padded prompt width: {padded_width}")

    all_match = True
    for index, prompt in enumerate(PROMPTS):
        padding_tokens = padded_width - prompt_lengths[index]
        row_matches = batched_token_ids[index] == independent_token_ids[index]
        all_match = all_match and row_matches

        print()
        print(f"Request {index}")
        print(f"    Prompt: {prompt}")
        print(f"    Real prompt tokens {prompt_lengths[index]}")
        print(f"    Left-padding tokens {padding_tokens}")
        print(f"    Attention mask {batch_attention_mask[index].tolist()}")
        print(f"    Position IDs {batch_position_ids[index].tolist()}")
        print(f"    Batched token IDs {batched_token_ids[index]}")
        print(f"    Independent token ids {independent_token_ids[index]}")
        print(f"    Next ones {tokenizer.decode(batch_generated_ids[index])}")
        print(f"    Matches independent generation {row_matches}")

    if not all_match:
        raise RuntimeError(
            "At least 1 batch continuation differed from indep generation"
        )

    print()
    print("Every batched continuation matches indep generation")


if __name__ == "__main__":
    main()
        