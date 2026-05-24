import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

with open("multilingual/data/sft/v9/mixed/train_mixed.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r.get("metadata", {}).get("distilled"):
            continue  # skip distilled, find a raw one
        break

print("=== TRAINING FORMAT (full messages) ===")
print(tokenizer.apply_chat_template(
    r["messages"],
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True,
))

print("\n=== INFERENCE FORMAT (user only) ===")
print(tokenizer.apply_chat_template(
    r["messages"][:2],  # system + user only
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
))