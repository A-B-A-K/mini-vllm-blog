"""Generrate reference output using HF's generation engine"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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


prompt_length = input_ids.shape[1]

with torch.inference_mode():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )


new_token_ids = generated_ids[0, prompt_length:]
new_token_id_list = new_token_ids.tolist()

print(f"Device: {device}")
print(f"Data type: {dtype}")
print(f"Input shape: {input_ids.shape}")
print(f"Prompt token IDs: {input_ids[0].tolist()}")
print(f"Generated token IDs: {new_token_id_list}")
print(f"Generated tokens: {tokenizer.convert_ids_to_tokens(new_token_id_list)}")
print(f"Continuation: {tokenizer.decode(new_token_ids)}")