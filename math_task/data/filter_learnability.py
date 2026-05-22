"""Filter the RL problem pool to the learnability band [k_lo, k_hi] / N.

GRPO learns from the variance inside a rollout group. A problem where all
rollouts succeed gives advantage = 0; a problem where all fail also gives
advantage = 0. The signal is concentrated in the middle band (LIMR's LIM
score, DAPO's dynamic sampling, LIMO's "solved 1-3 of 32" rule).

This script runs N (default 8) rollouts of the BASE model on every problem
in the prepared JSONL, scores each completion with Math-Verify on the
`\\boxed{}` answer, and writes a filtered JSONL containing only problems
whose pass count is in [k_lo, k_hi] (default [1, 6] of 8).

The filter pass is one-shot — run it once before training starts. Cost on
a 1xA100 40GB with vLLM at 4K context, 8 completions/problem:
  ~50 problems/min  ->  ~13 hours for the full 40K DeepScaleR pool.
Use --limit during smoke-tests.

Usage:
    python -m math_task.data.filter_learnability \\
        --input  /scratch/<user>/rl_data/deepscaler_full.jsonl \\
        --output /scratch/<user>/rl_data/deepscaler_filtered.jsonl

    # Smoke test on 200 problems:
    python -m math_task.data.filter_learnability \\
        --input  /scratch/<user>/rl_data/deepscaler_full.jsonl \\
        --output /scratch/<user>/rl_data/deepscaler_filtered_smoke.jsonl \\
        --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List


DEFAULT_N_SAMPLES = 8
DEFAULT_K_LO = 1
DEFAULT_K_HI = 6
DEFAULT_MAX_TOKENS = 4096    # filter runs at 4K to be cheap; phase-1 also at 4K
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20


def _load_input(path: Path, limit: int | None) -> List[Dict]:
    """Read prepare_rl.py output and adapt to the {prompt, answer} schema that
    run_generation expects."""
    items: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            items.append({
                "prompt": rec["messages"][0]["content"],
                "answer": rec["solution"],
                "source": rec.get("source"),
            })
            if limit is not None and len(items) >= limit:
                break
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="Prepared RL JSONL from prepare_rl.py.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Filtered JSONL output.")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="Base model to use for the filter rollouts.")
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    ap.add_argument("--k-lo", type=int, default=DEFAULT_K_LO,
                    help=f"Minimum pass count (inclusive) to keep. Default {DEFAULT_K_LO}.")
    ap.add_argument("--k-hi", type=int, default=DEFAULT_K_HI,
                    help=f"Maximum pass count (inclusive) to keep. Default {DEFAULT_K_HI}.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="Per-completion max tokens (also caps vLLM max_model_len).")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke test: cap N problems.")
    ap.add_argument("--save-rollouts", type=Path, default=None,
                    help="If given, also write the raw rollouts as JSONL "
                         "(useful for inspecting failure modes / SFT bootstrap).")
    args = ap.parse_args()

    # Lazy imports.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from evaluate.score import score_generations

    items = _load_input(args.input, args.limit)
    print(f"[load] {len(items)} problems from {args.input}")

    # Build chat-templated prompts (thinking ON, matches inference contract).
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for it in items:
        messages = [{"role": "user", "content": it["prompt"]}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    print(f"[gen]  vLLM {args.model}  n={args.n_samples}  max_tokens={args.max_tokens}")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_tokens,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    outputs = llm.generate(prompts, sp)
    assert len(outputs) == len(items)

    # Score each problem and bucket by pass count.
    score_inputs = []
    for it, out in zip(items, outputs):
        score_inputs.append({
            "prompt": it["prompt"],
            "answer": it["answer"],
            "completions": [c.text for c in out.outputs],
        })

    scored = score_generations(score_inputs, method="boxed")
    # score_generations returns:
    #   {benchmark_method, n_problems, n_completions, metrics,
    #    detailed_results: [{index, prompt, reference, n, c, completions: [{extracted, correct}]}]}
    per_problem = scored.get("detailed_results") or []
    if not per_problem:
        raise RuntimeError(
            "score_generations returned no detailed_results. "
            f"Got keys: {list(scored.keys())}"
        )
    print(f"[score] overall metrics on these {len(items)} problems: {scored.get('metrics')}")

    # ---- 1. Persist rollouts FIRST so a downstream bug can never lose them.
    if args.save_rollouts:
        args.save_rollouts.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_rollouts, "w") as rollouts_fp:
            for it, prob, out in zip(items, per_problem, outputs):
                comp_details = prob.get("completions") or []
                rollouts_fp.write(json.dumps({
                    "prompt": it["prompt"],
                    "answer": it["answer"],
                    "source": it.get("source"),
                    "n_pass": int(prob.get("c", 0)),
                    "n_sampled": args.n_samples,
                    "completions": [c.text for c in out.outputs],
                    "completion_correct": [bool(cd.get("correct")) for cd in comp_details],
                    "extracted": [cd.get("extracted") for cd in comp_details],
                }, ensure_ascii=False) + "\n")
        print(f"[save] rollouts -> {args.save_rollouts}")

    # ---- 2. Apply the learnability band and write the filtered JSONL.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    pass_count_hist: Counter = Counter()
    with open(args.output, "w") as f:
        for it, prob in zip(items, per_problem):
            n_pass = int(prob.get("c", 0))
            pass_count_hist[n_pass] += 1
            if args.k_lo <= n_pass <= args.k_hi:
                rec = {
                    "messages": [{"role": "user", "content": it["prompt"]}],
                    "solution": it["answer"],
                    "source": it.get("source"),
                    "n_pass_base": n_pass,
                    "n_sampled_base": args.n_samples,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"\n[filter] pass-count histogram (out of {args.n_samples}):")
    max_count = max(pass_count_hist.values()) if pass_count_hist else 1
    for k in range(args.n_samples + 1):
        n_k = pass_count_hist.get(k, 0)
        bar = "#" * int(60 * n_k / max(1, max_count))
        print(f"  k={k:2d}  n={n_k:6d}  {bar}")
    print(f"\n[filter] kept {kept}/{len(items)} problems "
          f"(band [{args.k_lo}, {args.k_hi}])")
    print(f"[save]   -> {args.output}")


if __name__ == "__main__":
    main()
