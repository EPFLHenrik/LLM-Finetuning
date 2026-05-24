import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
path = "multilingual/data/sft/v9/mixed/train_mixed.jsonl"
out  = "long_examples_gt1024.txt"
THRESHOLD = 1600

results = []
with open(path) as f:
    for line in f:
        r = json.loads(line)
        text_rendered = tokenizer.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        n_tokens = len(tokenizer(text_rendered)["input_ids"])
        if n_tokens > THRESHOLD:
            results.append((n_tokens, r["id"], r["dataset"], r["language"], text_rendered))

results.sort(key=lambda x: -x[0])
print(f"Found {len(results)} examples > {THRESHOLD} tokens")

with open(out, "w", encoding="utf-8") as f:
    for n_tokens, rid, dataset, language, rendered in results:
        f.write(f"{'='*80}\n")
        f.write(f"id={rid}  dataset={dataset}  language={language}  tokens={n_tokens}\n")
        f.write(f"{'─'*80}\n")
        f.write(rendered + "\n\n")

print(f"Saved to {out}")