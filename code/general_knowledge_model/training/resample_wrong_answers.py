#!/usr/bin/env python3
"""Resample only incorrect answer rows using vLLM.

This script reads a JSONL/JSON file where rows already contain a `correct` flag.
Rows with `correct == false` are regenerated with the same chat/template logic as
`training/generate_answers.py`, then rescored. Rows marked correct are copied
through unchanged.
"""

from __future__ import annotations

import argparse
import json
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


TEACHER_MODEL_DIR = Path(__file__).resolve().parents[3] / "checkpoints" / "general_knowledge_model" / "teacher"

SYSTEM_MESSAGE = (
    "You are an expert teacher helping a much smaller, 1.7B parameter language model learn how to reason. "
    "Solve the following multiple-choice question. Inside your <think> tags, you must use simple, direct sentences, "
    "basic vocabulary, and short, linear logical steps. Do not write a PhD-level essay. Explain it as if you are "
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


def _build_messages(prompt: str, args: argparse.Namespace) -> list[dict[str, str]]:
    if args.student:
        return [
            {"role": "system", "content": SYSTEM_MESSAGE_STUDENT},
            {"role": "user", "content": prompt},
        ]
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]


def _format_prompt(tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
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


def _generate_completions_vllm(llm: LLM, prompts: list[str], sampling_params: SamplingParams, chunk_size: int = 0) -> list[str]:
    if chunk_size <= 0 or len(prompts) <= chunk_size:
        batches = [prompts]
    else:
        batches = [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]

    completions: list[str] = []
    for batch_idx, batch in enumerate(batches):
        if len(batches) > 1:
            print(f"Generating chunk {batch_idx + 1}/{len(batches)} ({len(batch)} prompts)...")
        outputs = llm.generate(batch, sampling_params=sampling_params)
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


def _is_marked_incorrect(row: dict[str, Any]) -> bool:
    value = row.get("correct")
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no"}:
            return True
        if normalized in {"true", "1", "yes"}:
            return False
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resample only rows with correct=false")
    parser.add_argument("input", type=Path, help="Input JSON or JSONL file with prompt/answer/correct rows")
    parser.add_argument("--hf-token", required=True, help="Hugging Face access token")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <input>.resampled.jsonl)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=TEACHER_MODEL_DIR,
        help="Local model checkpoint directory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for vLLM")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel degree")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization target")
    parser.add_argument("--dtype", default="auto", help="vLLM dtype, for example auto, float16, or bfloat16")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in FP8 quantization via vLLM (reduces VRAM usage, no bitsandbytes needed)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override vLLM max_model_len to reduce KV cache memory usage")
    parser.add_argument("--max-num-seqs", type=int, default=32, help="Maximum number of sequences vLLM processes concurrently (lower = less activation memory)")
    parser.add_argument("--student", action="store_true", help="Student model")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Split prompts into chunks of this size before passing to vLLM to reduce peak KV-cache memory (0 = no chunking)",
    )
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path.with_suffix(".resampled.jsonl")
    model_dir = args.model_dir.expanduser().resolve()

    rows = _read_json_input(input_path)
    if not rows:
        raise SystemExit(f"{input_path}: no rows found")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    wrong_indices = [i for i, row in enumerate(rows) if _is_marked_incorrect(row)]
    if not wrong_indices:
        print("No rows marked with correct=false. Writing passthrough output.")
        _write_jsonl(output_path, rows)
        print(f"Wrote {output_path}")
        return 0

    for i in wrong_indices:
        row = rows[i]
        if "prompt" not in row or "answer" not in row:
            raise SystemExit(f"Row {i + 1}: expected 'prompt' and 'answer' for resampling")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=args.hf_token, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [_format_prompt(tokenizer, str(rows[i]["prompt"]), args) for i in wrong_indices]

    print(f"Loading vLLM model from {model_dir}...")
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
        llm_kwargs["quantization"] = "fp8"
        llm_kwargs["enforce_eager"] = True
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = _build_sampling_params(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    completions = _generate_completions_vllm(llm, prompts, sampling_params, chunk_size=args.chunk_size)

    updated = 0
    newly_correct = 0
    for position, row_index in enumerate(wrong_indices, start=1):
        row = dict(rows[row_index])
        completion = completions[position - 1]
        reference = str(row["answer"])
        extracted, correct, reason, strategy = _score_completion(completion, reference)

        # Only accept the teacher-generated completion if it is judged correct.
        # If the teacher is still wrong, preserve the original row and record
        # the teacher attempt in separate fields so we don't overwrite a
        # previously-correct student answer with a wrong teacher answer.
        if correct:
            row["completion"] = completion
            row["extracted_answer"] = extracted
            row["correct"] = True
            row["reason"] = reason
            row["extraction_strategy"] = strategy
            row["system_message"] = SYSTEM_MESSAGE_STUDENT if args.student else SYSTEM_MESSAGE
            newly_correct += 1
            print(
                f"[{position}/{len(wrong_indices)}] row={row_index + 1} "
                f"accepted teacher correct={correct} reason={reason} extracted={extracted!r}"
            )
        else:
            # Keep the original completion/correct flag; store teacher's
            # attempt under teacher_* keys for later inspection.
            row["teacher_completion"] = completion
            row["teacher_extracted_answer"] = extracted
            row["teacher_correct"] = False
            row["teacher_reason"] = reason
            row["teacher_extraction_strategy"] = strategy
            row["teacher_system_message"] = SYSTEM_MESSAGE_STUDENT if args.student else SYSTEM_MESSAGE
            print(
                f"[{position}/{len(wrong_indices)}] row={row_index + 1} "
                f"teacher_incorrect kept_original reason={reason} extracted={extracted!r}"
            )

        rows[row_index] = row
        updated += 1

    _write_jsonl(output_path, rows)

    final_correct = sum(1 for row in rows if bool(row.get("correct")))
    print(f"Wrote {output_path}")
    print(f"Resampled rows: {updated}")
    print(f"Now correct among resampled rows: {newly_correct}/{updated} ({(newly_correct / updated):.2%})")
    print(f"Overall correct rows in output: {final_correct}/{len(rows)} ({(final_correct / len(rows)):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
