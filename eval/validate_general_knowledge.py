#!/usr/bin/env python3
"""Validate the general knowledge snapshot with the local baseline model.

The script loads `models/qwen3_baseline`, reads
`validation_samples/general_knowledge.jsonl`, runs generation for each prompt,
and reports exact-match accuracy against boxed answers.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen3_thinking"
DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "validation_samples" / "general_knowledge.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "general_knowledge_validation.jsonl"


def load_samples(path: Path, limit: int | None = None):
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def extract_boxed(text: str) -> str | None:
    match = BOX_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_answer_ci_style(
    model,
    tokenizer,
    formatted_prompt: str,
    generation_config: GenerationConfig,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    local_config = copy.deepcopy(generation_config)
    local_config.max_new_tokens = max_new_tokens
    if local_config.pad_token_id is None:
        local_config.pad_token_id = tokenizer.eos_token_id
    if local_config.eos_token_id is None:
        local_config.eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output = model.generate(**inputs, generation_config=local_config)

    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation on the general knowledge snapshot.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Path to the local checkpoint directory.")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="Path to the general knowledge JSONL file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to write detailed results JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of examples to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum number of generated tokens per example.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    samples_path = Path(args.samples)

    print(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_config = GenerationConfig.from_pretrained(model_dir)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model from {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype)
    model.eval()

    if args.device == "cpu":
        model = model.to("cpu")
    elif args.device == "cuda":
        model = model.to("cuda")
    elif torch.cuda.is_available():
        model = model.to("cuda")

    samples = load_samples(samples_path, limit=args.limit)
    total = len(samples)
    correct = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle = output_path.open("w", encoding="utf-8")

    try:
        for index, sample in enumerate(samples, start=1):
            prompt = sample["prompt"]
            expected = str(sample["answer"]).strip()
            formatted_prompt = format_prompt(tokenizer, prompt)
            completion = generate_answer_ci_style(
                model,
                tokenizer,
                formatted_prompt,
                generation_config,
                args.max_new_tokens,
            )
            boxed = extract_boxed(completion)
            is_correct = boxed is not None and boxed.strip() == expected
            correct += int(is_correct)

            record = {
                "index": index,
                "source_file": samples_path.name,
                "prompt": prompt,
                "gold_answer": expected,
                "formatted_prompt": formatted_prompt,
                "actual_answer": completion.strip(),
                "boxed_answer": boxed,
                "correct": is_correct,
            }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("---")
            print(f"Example {index}/{total}")
            print(f"Expected: {expected}")
            print(f"Boxed: {boxed}")
            print(f"Correct: {is_correct}")
            print("Completion:")
            print(completion.strip())
    finally:
        output_handle.close()

    accuracy = 0.0 if total == 0 else correct / total
    print("===")
    print(f"Summary: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Wrote detailed results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())