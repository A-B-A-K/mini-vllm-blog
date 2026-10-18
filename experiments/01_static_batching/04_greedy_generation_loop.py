"""Reproduce HF generation with an explicit greed decode loop"""

from mini_vllm.generation import generate_greedy
from mini_vllm.model_loader import load_model_and_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 4
EXPECTED_TOKEN_IDS = [12095, 13, 1084, 374]


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, MODEL_REVISION)

    device = model.device

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    generated_ids = generate_greedy(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    actual_token_ids = generated_ids[0].tolist()

    print(f"Device {device}")
    print(f"data type {model.dtype}")
    print(f"prompt token IDS {input_ids[0].tolist()}")
    print(f"generated token ids {actual_token_ids}")
    print(
        f"generated tokens {tokenizer.convert_ids_to_tokens(actual_token_ids)}"
    )
    print(f"continuation {tokenizer.decode(generated_ids[0])}")

    if actual_token_ids != EXPECTED_TOKEN_IDS:
        raise RuntimeError(
            f"Expected reference tokens {EXPECTED_TOKEN_IDS}, "
            f"got instead {actual_token_ids}"
        )

    print("Matches the complete HF reference: yes")


if __name__ == "__main__":
    main()