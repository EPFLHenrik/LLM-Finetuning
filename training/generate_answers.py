#!/usr/bin/env python3
"""Generate teacher answers for prompt/answer records using the local Qwen checkpoint.

The script reads a JSON or JSONL file containing objects with at least `prompt`
and `answer`, sends the prompt through the teacher model in
`models/teacher`, and writes a JSONL file with the generated completion,
extracted answer, and correctness flag.

Only the `prompt` and `answer` fields are used from the input rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.benchmarks import extract_benchmark_answer, is_correct_benchmark_answer
from evaluate.extract_answer import extract_boxed_answer_with_fallbacks

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise SystemExit("vLLM is not installed in this environment. Install vllm to use this script.") from exc


TEACHER_MODEL_DIR = Path("/scratch/workspace/henrik/standard-project-m2-mystery-machine/models/teacher")

SYSTEM_MESSAGE = (
    "You are an expert teacher helping a much smaller, 1.7B parameter language model learn how to reason. "
    "Solve the following multiple-choice question. Inside your <think> tags, provide a solid Chain of Thought that helps the model. "
    "Explain it as if you are "
    "leaving a concise breadcrumb trail for a smaller model to follow. End with the final answer in \\boxed{} so "
    "for example \\boxed{A} if A is the correct answer!"
)

SYSTEM_MESSAGE_STUDENT = (
    "You are a careful reasoning assistant.\n\n"
    "Reason through the problem when needed.\n\n"
    "Your final answer must appear exactly once inside \\boxed{...}.\n\n"
    "Rules:\n"
    "- For multiple-choice questions, output only the correct capital letter inside \\boxed{...}.\n"
    "- For math questions, output only the final answer inside \\boxed{...}.\n"
    "- Do not include explanations or extra text inside the box.\n"
    "- Never output multiple boxed answers.\n"
    "- Never leave the final answer unboxed.\n\n"
    "Examples:\n"
    "\\boxed{B}\n"
    "\\boxed{42}"
)


def _read_json_input(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{lineno}: each row must be a JSON object")
                rows.append(row)
        return rows

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        if not all(isinstance(row, dict) for row in payload):
            raise SystemExit(f"{path}: JSON array must contain only objects")
        return list(payload)

    if isinstance(payload, dict):
        for key in ("data", "items", "examples", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                if not all(isinstance(row, dict) for row in value):
                    raise SystemExit(f"{path}: '{key}' must contain only objects")
                return list(value)

    raise SystemExit(
        f"{path}: expected a JSON array, a JSON object with a 'data'/'items'/'examples'/'rows' list, or JSONL"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_messages(prompt: str, args) -> list[dict[str, str]]:
    if args.student:
        return [
            {"role": "system", "content": SYSTEM_MESSAGE_STUDENT},
            {"role": "user", "content": prompt},
        ]
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]


def _format_prompt(tokenizer, prompt: str, args) -> str:
    messages = _build_messages(prompt, args)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _build_sampling_params(*, max_new_tokens: int, temperature: float, top_p: float, top_k: int, seed: int | None) -> SamplingParams:
    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )


def _generate_completions_vllm(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    completions: list[str] = []
    for request_output in outputs:
        if request_output.outputs:
            completions.append(request_output.outputs[0].text.strip())
        else:
            completions.append("")
    return completions


def _choose_extraction_method(reference: str) -> str:
    if re.fullmatch(r"[A-Z]", reference.strip().upper()):
        return "knowledge"
    return "boxed"


def _score_completion(completion: str, reference: str) -> tuple[str | None, bool, str, str]:
    reference_text = reference.strip().upper()
    if re.fullmatch(r"[A-Z]", reference_text):
        extracted, strategy = extract_boxed_answer_with_fallbacks(completion)
        if extracted is None:
            return extracted, False, "no_answer_found", strategy
        if extracted == reference_text and strategy == "boxed_exact_capital":
            return extracted, True, "correct", strategy
        if extracted == reference_text:
            return extracted, True, strategy, strategy
        return extracted, False, "wrong answer", strategy

    method = _choose_extraction_method(reference)
    extracted = extract_benchmark_answer(completion, method, reference)
    correct = is_correct_benchmark_answer(extracted, reference, method)
    reason = "correct" if correct else "wrong answer"
    return extracted, correct, reason, method


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and score teacher answers")
    parser.add_argument("input", type=Path, help="Input JSON or JSONL file with prompt/answer rows")
    parser.add_argument("--hf-token", required=True, help="Hugging Face access token")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <input>.teacher_answers.jsonl)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=TEACHER_MODEL_DIR,
        help="Local teacher checkpoint directory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for vLLM")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel degree")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization target")
    parser.add_argument("--dtype", default="auto", help="vLLM dtype, for example auto, float16, or bfloat16")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit via bitsandbytes (reduces VRAM usage)")
    parser.add_argument("--student", action="store_true", help="Student model")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start generating from this input row index and append results to the output file",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save progress to the output file every N prompts (0 to disable)",
    )
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path.with_suffix(".teacher_answers.jsonl")
    model_dir = args.model_dir.expanduser().resolve()

    rows = _read_json_input(input_path)
    if not rows:
        raise SystemExit(f"{input_path}: no rows found")
    if args.start_index < 0 or args.start_index > len(rows):
        raise SystemExit(f"--start-index must be between 0 and {len(rows)}")

    if args.start_index:
        rows = rows[args.start_index :]

    if not model_dir.exists():
        raise FileNotFoundError(f"Teacher model directory not found: {model_dir}")

    hf_token = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=hf_token, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_prompts = [
        _format_prompt(tokenizer, str(row["prompt"]), args)
        for row in rows
    ]

    print(f"Loading vLLM teacher model from {model_dir}...")
    llm_kwargs: dict = dict(
        model=str(model_dir),
        tokenizer=str(model_dir),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
    )
    if args.load_in_8bit:
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"
    llm = LLM(**llm_kwargs)
    sampling_params = _build_sampling_params(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    completions = _generate_completions_vllm(llm, formatted_prompts, sampling_params)

    results: list[dict[str, Any]] = []
    correct_count = 0
    save_every = int(args.save_every) if getattr(args, "save_every", None) is not None else 0
    append_mode = args.start_index > 0
    last_saved_count = 0

    if append_mode:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)

    for index, (row, completion) in enumerate(zip(rows, completions), start=1):
        if "prompt" not in row or "answer" not in row:
            raise SystemExit(f"Row {index}: expected both 'prompt' and 'answer' fields")

        prompt = str(row["prompt"])
        reference = str(row["answer"])
        extracted, correct, reason, strategy = _score_completion(completion, reference)
        correct_count += int(correct)

        results.append(
            {
                "prompt": prompt,
                "answer": reference,
                "completion": completion,
                "extracted_answer": extracted,
                "correct": correct,
                "reason": reason,
                "extraction_strategy": strategy,
                "system_message": SYSTEM_MESSAGE_STUDENT if args.student else SYSTEM_MESSAGE,
            }
        )

        print(f"[{index}/{len(rows)}] correct={correct} reason={reason} extracted={extracted!r}")

        # Periodically save progress so long runs can be resumed / inspected.
        if save_every and index % save_every == 0:
            new_rows = results[last_saved_count:]
            if append_mode:
                _append_jsonl(output_path, new_rows)
            else:
                _write_jsonl(output_path, results)
            last_saved_count = len(results)
            print(f"Checkpoint: wrote progress at {index} rows to {output_path}")

    remaining_rows = results[last_saved_count:]
    if append_mode:
        _append_jsonl(output_path, remaining_rows)
    else:
        _write_jsonl(output_path, results)
    accuracy = correct_count / len(results)
    print(f"Wrote {output_path}")
    print(f"Accuracy: {correct_count}/{len(results)} ({accuracy:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())