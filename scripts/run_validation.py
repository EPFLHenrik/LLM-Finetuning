#!/usr/bin/env python3
"""Run local validation using the provided `validation_samples/` JSONL files.

This script attempts to load a local checkpoint (default: `models/qwen3_baseline`).
If the checkpoint is missing or you specify `--use-tiny`, it falls back to `gpt2`
so you can test the pipeline quickly.

Usage examples:
  python scripts/run_validation.py --model-dir models/qwen3_baseline
  python scripts/run_validation.py --use-tiny
"""
import argparse
import json
import os
import re
from glob import glob

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(text):
    m = BOX_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def load_validation_samples(root="validation_samples"):
    files = sorted(glob(os.path.join(root, "*.jsonl")))
    samples = []
    for f in files:
        with open(f, "r") as fh:
            for line in fh:
                data = json.loads(line)
                samples.append((os.path.basename(f), data["prompt"], data["answer"]))
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/qwen3_baseline")
    p.add_argument("--device", type=int, default=0, help="CUDA device id or -1 for CPU")
    p.add_argument("--use-tiny", action="store_true", help="Use gpt2 to test pipeline quickly")
    args = p.parse_args()

    model_name = "gpt2" if args.use_tiny else args.model_dir

    print("Loading tokenizer and model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = -1 if args.device < 0 else args.device
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    samples = load_validation_samples()
    total = 0
    correct = 0
    results = []

    for fname, prompt, answer in samples:
        total += 1
        # If tokenizer provides apply_chat_template, use it; otherwise use raw prompt
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        out = gen(formatted, max_new_tokens=128, do_sample=True, num_return_sequences=1)[0]["generated_text"]
        boxed = extract_boxed(out)
        ok = False
        if boxed is not None:
            ok = boxed.strip() == str(answer).strip()

        results.append((fname, prompt, answer, out, boxed, ok))
        if ok:
            correct += 1

    print(f"\nSummary: {correct}/{total} correct ({100.0 * correct/total:.1f}%)")

    # Print a few examples
    for r in results[:10]:
        fname, prompt, answer, out, boxed, ok = r
        print("---")
        print("file:", fname)
        print("prompt:", prompt)
        print("expected:", answer)
        print("boxed:", boxed)
        print("ok:", ok)


if __name__ == "__main__":
    main()
