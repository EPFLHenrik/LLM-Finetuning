"""Distill chain-of-thought traces from a strong teacher model.

For competition-math datasets that ship problems + gold answers but no
reasoning traces (AIME, AMC), generate `<think>...</think>...\\boxed{...}`
traces using a quantized DeepSeek-R1-Distill-Qwen teacher, then
rejection-sample on Math-Verify correctness.

Input JSONL schema (one per line):
    {"problem": str, "gold": str, "source": str}

Output JSONL schema:
    {"problem": str, "gold": str, "source": str,
     "traces": [str, str, ...],   # 1-4 correct traces, longest first
     "n_attempted": int, "n_correct": int}

Usage:
    python -m code.math_model.preprocessing.sft.distill_traces \\
        --input  data/math/sft/workdir/needs_distill.jsonl \\
        --output data/math/sft/workdir/distilled.jsonl \\
        --teacher Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ \\
        --n-samples 8 --keep-top-k 4 --max-tokens 8192

Notes:
- AWQ quantization fits a 32B teacher on a single 40GB A100 (~36 GB used).
- vLLM detects AWQ via the model name suffix or the `quantization` flag.
- Resumes by skipping rows whose `problem` is already in the output file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


DEFAULT_TEACHER = "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ"
DEFAULT_N_SAMPLES = 8
DEFAULT_KEEP_TOP_K = 4
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95


def _load_jsonl(path: Path) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _already_done_problems(output_path: Path) -> set:
    """Resume support: return the set of problems already written to output."""
    if not output_path.exists():
        return set()
    seen = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("problem"):
                    seen.add(rec["problem"])
            except json.JSONDecodeError:
                continue
    return seen


def _filter_by_correctness(
    problem: str, gold: str, completions: List[str]
) -> List[str]:
    """Return only completions whose extracted `\\boxed{}` answer matches gold
    under the same logic the evaluator uses (`evaluation.answer_extraction`)."""
    from code.math_model.evaluation.answer_extraction import extract_boxed_answer, is_equiv

    correct: List[str] = []
    for comp in completions:
        extracted = extract_boxed_answer(comp, strip_double_curly_brace=True)
        if extracted is None:
            continue
        if is_equiv(str(extracted), str(gold)):
            correct.append(comp)
    return correct


def _has_think_block(text: str) -> bool:
    return "<think>" in text and "</think>" in text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="JSONL with rows {problem, gold, source}.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output JSONL with distilled traces. Resumable.")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER,
                    help=f"HF model id. Default {DEFAULT_TEACHER}.")
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    ap.add_argument("--keep-top-k", type=int, default=DEFAULT_KEEP_TOP_K,
                    help="Max correct traces to keep per problem. Longest first.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="Per-completion max output tokens (also caps vLLM max_model_len).")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap how many problems to distill (smoke test).")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Problems per vLLM generate() call. Each problem emits "
                         "n_samples completions, so the engine sees "
                         "batch_size * n_samples concurrent requests.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85,
                    help="vLLM --gpu-memory-utilization. 0.85 leaves room for "
                         "AWQ overhead on a 40GB A100.")
    args = ap.parse_args()

    items = _load_jsonl(args.input)
    if args.limit is not None:
        items = items[:args.limit]
    print(f"[load] {len(items)} problems from {args.input}")

    done = _already_done_problems(args.output)
    if done:
        print(f"[resume] skipping {len(done)} problems already in {args.output}")
        items = [it for it in items if it.get("problem") not in done]
        if not items:
            print("[resume] nothing left to do; exiting.")
            return

    # Lazy imports — keep --help fast without GPU/vllm.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    llm = LLM(
        model=args.teacher,
        max_model_len=args.max_tokens + 1024,   # add room for the prompt
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="auto",                            # AWQ quants pick their own dtype
        trust_remote_code=False,
    )
    sp = SamplingParams(
        n=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Open output in append mode so resume Just Works.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output, "a", buffering=1)  # line-buffered

    n_problems = 0
    n_with_any_correct = 0
    n_total_correct = 0
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start : batch_start + args.batch_size]

        # Build chat-templated prompts for the teacher.
        prompts = []
        for it in batch:
            messages = [{"role": "user", "content": it["problem"]}]
            prompts.append(
                tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            )

        outputs = llm.generate(prompts, sp)
        assert len(outputs) == len(batch)

        for it, out in zip(batch, outputs):
            completions = [c.text for c in out.outputs]
            # Filter to completions that contain a think block AND have correct boxed answer
            correct = [c for c in _filter_by_correctness(it["problem"], it["gold"], completions)
                       if _has_think_block(c)]
            # Keep longest correct traces first (more reasoning is usually richer).
            correct.sort(key=len, reverse=True)
            kept = correct[: args.keep_top_k]

            rec = {
                "problem": it["problem"],
                "gold": it["gold"],
                "source": it.get("source", "unknown"),
                "traces": kept,
                "n_attempted": args.n_samples,
                "n_correct": len(correct),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_problems += 1
            n_total_correct += len(correct)
            if kept:
                n_with_any_correct += 1

        # Heartbeat so we know it's still moving.
        print(
            f"[distill] processed={n_problems}/{len(items)}  "
            f"with_any_correct={n_with_any_correct}  "
            f"avg_correct/problem={n_total_correct / max(1, n_problems):.2f}"
        )

    out_f.close()
    print(f"\n[done] wrote {n_problems} problems to {args.output}")
    print(
        f"[done] retention: {n_with_any_correct}/{n_problems} "
        f"({n_with_any_correct / max(1, n_problems):.1%}) have >=1 correct trace"
    )


if __name__ == "__main__":
    main()
