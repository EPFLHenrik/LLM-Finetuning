"""multilingual/eval/run_eval_vllm.py

vLLM-based evaluation — drop-in replacement for run_eval.py.
Batches all prompts through vLLM's LLM engine for fast offline inference.

Usage:
    # Merged HF model
    python -m multilingual.eval.run_eval_vllm \
        --model_name cs-552-2026-mystery-machine/multilingual_model \
        --val_file multilingual/data/sft/v9/mixed/val_mixed.jsonl \
        --output_dir multilingual/results/eval/v9_vllm

    # Local merged model
    python -m multilingual.eval.run_eval_vllm \
        --model_name multilingual/results/merged/v13 \
        --val_file multilingual/data/benchmark/benchmark_2k.jsonl \
        --output_dir multilingual/results/eval/v13_benchmark

    # Provided examples only
    python -m multilingual.eval.run_eval_vllm \
        --model_name cs-552-2026-mystery-machine/multilingual_model \
        --provided_only \
        --output_dir multilingual/results/eval/v9_provided
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from vllm import LLM, SamplingParams

from multilingual.config import (
    DEFAULT_MAX_NEW_TOKENS_BY_TASK,
    DEFAULT_MODEL_NAME,
    REPO,
    SYSTEM_PROMPT,
)
from multilingual.utils import save_jsonl
from multilingual.eval.eval_utils import save_eval_summaries, save_summary_txt

CHAT_TEMPLATE = REPO / "multilingual" / "assets" / "chat_template.jinja"

# ---------------------------------------------------------------------------
# Answer extraction (mirrors evaluation.py logic)
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}."""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1).strip() if m else None


def _is_correct(pred: Optional[str], gold: str, task_type: str) -> bool:
    if pred is None:
        return False
    if task_type == "mcq":
        return pred.strip().upper() == gold.strip().upper()
    # math: normalise whitespace
    return pred.strip() == gold.strip()

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_chat(prompt: str, tokenizer, system_prompt: str) -> str:
    """Apply chat template to a single prompt string."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # thinking mode
    )

# ---------------------------------------------------------------------------
# Data helpers (identical to run_eval.py)
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _sft_to_eval_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []
    for r in records:
        prompt = r.get("prompt") or (
            r["messages"][1]["content"] if r.get("messages") else ""
        )
        examples.append({
            "id":        r.get("id", ""),
            "dataset":   r.get("dataset", "unknown"),
            "language":  r.get("language", "unknown"),
            "task_type": r.get("task_type", "mcq"),
            "prompt":    prompt,
            "answer":    r.get("answer", ""),
            "metadata":  r.get("metadata", {}),
        })
    return examples


def _provided_to_eval_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id":        f"provided_{i}",
            "dataset":   "provided",
            "language":  "mixed",
            "task_type": "mcq",
            "prompt":    r["prompt"],
            "answer":    r["answer"],
            "metadata":  {},
        }
        for i, r in enumerate(records)
    ]

# ---------------------------------------------------------------------------
# Core eval with vLLM
# ---------------------------------------------------------------------------

def run_vllm_eval(
    examples: List[Dict[str, Any]],
    llm: LLM,
    tokenizer,
    max_new_tokens_by_task: Dict[str, int],
    sampling_params_base: Dict,
) -> pd.DataFrame:
    """
    Run evaluation on a list of examples using vLLM batch inference.
    Returns a DataFrame with one row per example.
    """
    # Build prompts and per-example max_tokens
    formatted_prompts = []
    for ex in examples:
        chat = _build_chat(ex["prompt"], tokenizer, SYSTEM_PROMPT)
        formatted_prompts.append(chat)

    # Use max over task types for a single batch (or group by task if needed)
    task_types = [ex["task_type"] for ex in examples]
    max_tokens = max(max_new_tokens_by_task.get(t, 512) for t in task_types)

    sampling = SamplingParams(max_tokens=max_tokens, **sampling_params_base)
    outputs = llm.generate(formatted_prompts, sampling_params=sampling)

    rows = []
    for ex, out in zip(examples, outputs):
        generated = out.outputs[0].text
        pred       = _extract_boxed(generated)
        correct    = _is_correct(pred, ex["answer"], ex["task_type"])
        rows.append({
            "id":        ex["id"],
            "dataset":   ex["dataset"],
            "language":  ex["language"],
            "task_type": ex["task_type"],
            "prompt":    ex["prompt"],
            "gold":      ex["answer"],
            "generated": generated,
            "predicted": pred,
            "correct":   correct,
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--val_file",    type=Path, default=None)
    parser.add_argument("--output_dir",  type=Path,
                        default=REPO / "multilingual" / "results" / "eval")
    parser.add_argument("--n",           type=int, default=None,
                        help="Cap total examples from val_file.")
    parser.add_argument("--eval_provided",  action="store_true")
    parser.add_argument("--provided_only",  action="store_true")
    parser.add_argument("--provided_file",  type=Path,
                        default=REPO / "validation_samples" / "multilingual.jsonl")
    parser.add_argument("--no_custom_template", action="store_true")
    parser.add_argument("--chat_template", type=Path, default=None,
                        help="Path to custom chat template jinja file. Default: assets/chat_template.jinja.")
    parser.add_argument("--mcq_max_new_tokens",  type=int,
                        default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["mcq"])
    parser.add_argument("--math_max_new_tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["math"])
    parser.add_argument("--do_sample",   action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p",       type=float, default=0.95)
    parser.add_argument("--top_k",       type=int,   default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--overwrite",   action="store_true")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.perf_counter()
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = {
        "mcq":  args.mcq_max_new_tokens,
        "math": args.math_max_new_tokens,
    }

    print("=" * 60)
    print(f"Model:      {args.model_name}")
    print(f"Val file:   {args.val_file}")
    print(f"Output:     {args.output_dir}")
    print(f"TP size:    {args.tensor_parallel_size}")
    print("=" * 60)

    # Load tokenizer for chat template (vLLM handles model loading separately)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=hf_token, trust_remote_code=True,
    )
    template_path = args.chat_template or CHAT_TEMPLATE
    if not args.no_custom_template and template_path.exists():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    # Init vLLM engine
    llm = LLM(
        model=args.model_name,
        tokenizer=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )

    sampling_params_base = dict(
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p      if args.do_sample else 1.0,
        top_k=args.top_k      if args.do_sample else -1,
    )

    all_dfs = []

    # ── Val file ──────────────────────────────────────────────────────────────
    if not args.provided_only and args.val_file is not None:
        out_path = args.output_dir / "val_results.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} — use --overwrite to re-run")
            all_dfs.append(pd.read_json(out_path, lines=True))
        else:
            records  = _load_jsonl(args.val_file)
            if args.n:
                records = records[:args.n]
            examples = _sft_to_eval_examples(records)
            print(f"Loaded {len(examples)} examples from {args.val_file.name}")

            df_val = run_vllm_eval(
                examples, llm, tokenizer, max_new_tokens, sampling_params_base,
            )
            save_jsonl(df_val, out_path)
            all_dfs.append(df_val)

    # ── Provided examples ─────────────────────────────────────────────────────
    if args.eval_provided or args.provided_only:
        out_path = args.output_dir / "provided_results.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} — use --overwrite to re-run")
            all_dfs.append(pd.read_json(out_path, lines=True))
        elif not args.provided_file.exists():
            print(f"[WARN] Provided file not found: {args.provided_file}")
        else:
            records  = _load_jsonl(args.provided_file)
            examples = _provided_to_eval_examples(records)
            print(f"Evaluating {len(examples)} provided examples...")
            df_prov = run_vllm_eval(
                examples, llm, tokenizer, max_new_tokens, sampling_params_base,
            )
            save_jsonl(df_prov, out_path)
            all_dfs.append(df_prov)

    # ── Summary ───────────────────────────────────────────────────────────────
    if not all_dfs:
        print("No results produced.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # summary
    save_jsonl(df_all, args.output_dir / "all_results.jsonl")
    summaries = save_eval_summaries(df_all, args.output_dir)

    summary_by_dataset_language = summaries["summary_by_dataset_language"]
    summary_by_language = summaries["summary_by_language"]
    summary_by_dataset = summaries["summary_by_dataset"]
    overall = summaries["overall"]
    paths = summaries["paths"]

    runtime_seconds = time.perf_counter() - start_time

    summary_txt_path = save_summary_txt(
        model_name=args.model_name,
        output_dir=args.output_dir,
        summary_by_dataset_language=summary_by_dataset_language,
        summary_by_language=summary_by_language,
        summary_by_dataset=summary_by_dataset,
        overall=overall,
        runtime_seconds=runtime_seconds,
    )

    print("\n" + "=" * 60)
    print("SUMMARY (per dataset/language)")
    print("=" * 60)
    print(summary_by_dataset_language.to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY (per language)")
    print("=" * 60)
    print(summary_by_language.to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY (per dataset)")
    print("=" * 60)
    print(summary_by_dataset.to_string(index=False))

    print(f"\nOverall accuracy: {overall['accuracy']:.3f} ({overall['correct']}/{overall['n']})")
    print(f"Runtime seconds: {runtime_seconds:.2f}")
    print(f"Runtime minutes: {runtime_seconds / 60:.2f}")

    print("\nSummary files saved:")
    for path in paths.values():
        print(f"  {path}")
    print(f"  {summary_txt_path}")


if __name__ == "__main__":
    main()