#!/usr/bin/env python3
"""Validate the general knowledge snapshot with a local checkpoint using vLLM.

This is the vLLM-backed sibling of `validate_general_knowledge.py`. It keeps the
same sample loading, chat-template formatting, boxed-answer extraction, and
result logging, but runs batched inference through vLLM instead of
`model.generate(...)`.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise SystemExit("vLLM is not installed in this environment. Install vllm to use this script.") from exc


BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen3_thinking"
DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "validation_samples" / "general_knowledge.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "general_knowledge_validation_vllm.jsonl"


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
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=None)
    return prompt


def build_sampling_params(model_dir: Path, max_new_tokens: int) -> SamplingParams:
    generation_config = GenerationConfig.from_pretrained(model_dir)
    local_config = copy.deepcopy(generation_config)

    temperature = 0.0 if local_config.temperature is None else float(local_config.temperature)
    top_p = 1.0 if local_config.top_p is None else float(local_config.top_p)
    top_k = -1 if local_config.top_k is None else int(local_config.top_k)

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
    )


def generate_answers_vllm(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    completions: list[str] = []
    for request_output in outputs:
        if request_output.outputs:
            completions.append(request_output.outputs[0].text)
        else:
            completions.append("")
    return completions


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation on the general knowledge snapshot using vLLM.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Path to the local checkpoint directory.")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="Path to the general knowledge JSONL file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to write detailed results JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of examples to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum number of generated tokens per example.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel degree.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3, help="vLLM GPU memory utilization target.")
    parser.add_argument("--dtype", default="auto", help="vLLM dtype, for example auto, float16, or bfloat16.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    samples_path = Path(args.samples)

    print(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)

    print(f"Loading vLLM model from {model_dir}")
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )

    samples = load_samples(samples_path, limit=args.limit)
    total = len(samples)
    if total == 0:
        raise SystemExit(f"No validation samples found in {samples_path}.")

    formatted_prompts = [format_prompt(tokenizer, sample["prompt"]) for sample in samples]
    sampling_params = build_sampling_params(model_dir, args.max_new_tokens)
    completions = generate_answers_vllm(llm, formatted_prompts, sampling_params)

    correct = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for index, (sample, formatted_prompt, completion) in enumerate(zip(samples, formatted_prompts, completions), start=1):
            prompt = sample["prompt"]
            expected = str(sample["answer"]).strip()
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

    accuracy = 0.0 if total == 0 else correct / total
    print("===")
    print(f"Summary: {correct}/{total} correct ({accuracy:.1%})")
    print(f"Wrote detailed results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())