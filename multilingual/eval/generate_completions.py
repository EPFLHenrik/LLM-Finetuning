"""multilingual/eval/generate_completions.py

Generates n completions per question and writes output compatible with
evaluate/score.py:
    {"prompt": "...", "answer": "A", "completions": ["...", ...]}

Usage:
    python -m multilingual.eval.generate_completions \
        --model_name Qwen/Qwen3-1.7B \
        --dataset include \
        --language russian \
        --output_path multilingual/results/gens/include_russian.jsonl \
        --n 8

Then score with:
    python -m evaluate.score \
        --generations multilingual/results/gens/include_russian.jsonl \
        --benchmark multilingual
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from tqdm.auto import tqdm

from multilingual.config import (
    DEFAULT_MAX_NEW_TOKENS_BY_TASK,
    DEFAULT_MODEL_NAME,
    REPO,
    SYSTEM_PROMPT,
    TARGET_LANGUAGES,
)
from multilingual.generation import load_model_and_tokenizer
from multilingual.loaders.ceval_loader import load_ceval_examples
from multilingual.loaders.include_loader import load_include_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples
from multilingual.loaders.mmmlu_loader import load_mmmlu_examples

DATASET_CHOICES = ["include", "mgsm", "mmmlu", "milu", "ceval"]


def load_examples(dataset: str, language: str, n_examples: Optional[int], hf_token: str):
    """Load val examples for a given dataset/language."""
    if dataset == "include":
        return load_include_examples(language, n=n_examples, split="val")
    if dataset == "mgsm":
        return load_mgsm_examples(language, n=n_examples, split="val")
    if dataset == "mmmlu":
        return load_mmmlu_examples(language, n=n_examples, split="val")
    if dataset == "milu":
        if language != "hindi":
            raise ValueError("MILU is Hindi-only.")
        return load_milu_examples(n=n_examples, split="val", hf_token=hf_token)
    if dataset == "ceval":
        if language != "chinese":
            raise ValueError("C-Eval is Chinese-only.")
        return load_ceval_examples(n=n_examples, split="val")
    raise ValueError(f"Unknown dataset: {dataset}")


@torch.inference_mode()
def generate_n_completions(
    model,
    tokenizer,
    prompt: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    """Generate n independent completions for one prompt."""
    from multilingual.generation import build_messages

    messages = build_messages(prompt)
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    # Generate all n completions in one batched call
    input_ids = inputs["input_ids"].expand(n, -1)
    attention_mask = inputs["attention_mask"].expand(n, -1)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    return [
        tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=False)
        for i in range(n)
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--language", required=True, choices=list(TARGET_LANGUAGES.keys()))
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--n", type=int, default=8, help="Completions per question.")
    parser.add_argument("--n_examples", type=int, default=None, help="Cap on val examples (None=full val set).")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["mcq"])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    model, tokenizer = load_model_and_tokenizer(args.model_name, hf_token)

    examples = load_examples(args.dataset, args.language, args.n_examples, hf_token)
    if not examples:
        print(f"No examples found for {args.dataset}/{args.language}.")
        return

    print(f"Generating {args.n} completions for {len(examples)} examples "
          f"({args.dataset}/{args.language})")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ex in tqdm(examples):
            completions = generate_n_completions(
                model=model,
                tokenizer=tokenizer,
                prompt=ex["prompt"],
                n=args.n,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            row = {
                "prompt":      ex["prompt"],
                "answer":      ex["answer"],
                "completions": completions,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved: {args.output_path}")
    print(f"Score with: python -m evaluate.score "
          f"--generations {args.output_path} --benchmark multilingual")


if __name__ == "__main__":
    main()