"""Predict one next token without using HF's generate method"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm.generation import prefill, select_greedy_tokens

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PROMPT = "The capital of France is"
EXPECTED_NEXT_TOKEN_ID = 12095


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=dtype,
    )

    model.to(device)
    model.eval()

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model_output = prefill(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    next_token_ids = select_greedy_tokens(model_output.logits)
    next_token_id = next_token_ids.item()

    print(f"{device = }")
    print(f"{dtype = }")
    print(f"{input_ids.shape = }")
    print(f"{model_output.logits.shape = }")
    print(f"{model_output.logits[:, -1, :].shape = }")
    print(f"{next_token_id = }")
    print(f"{tokenizer.convert_ids_to_tokens(next_token_id) = }")

    if next_token_id != EXPECTED_NEXT_TOKEN_ID:
        raise RuntimeError(
            f"Expected reference token {EXPECTED_NEXT_TOKEN_ID} \
            does not match {next_token_id}"
        )

    print("Matches HuggingFace reference: yes")


if __name__ == "__main__":
    main()
