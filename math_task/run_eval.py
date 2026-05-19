"""Entry point.

Run from the repo root:

    # 1-min smoke test on the 10 frozen validation problems (final budget = 16k)
    python -m math_task.run_eval --dataset validation

    # 100-problem stratified subset of MATH-500 (~30-45 min at 16k thinking ON)
    python -m math_task.run_eval --dataset math500_stratified --budget final

    # Full MATH-500 (~2-3 h at 16k thinking ON)
    python -m math_task.run_eval --dataset math500 --budget final

    # CI-budget run (4k) for comparison with the nightly EVAL_REPORT
    python -m math_task.run_eval --dataset math500_stratified --budget ci

    # Quick subset for iteration
    python -m math_task.run_eval --dataset math500 --budget final --limit 50

    # Score a finetune later
    python -m math_task.run_eval --dataset math500 --budget final --model cs-552-2026-<org>/math_model

    # Ablation cells (local-only knobs; CI ignores them):
    python -m math_task.run_eval --dataset math500 --limit 100 --prompt-variant boxed --thinking off

Produces three files in math_task/results/. Stem includes prompt-variant +
thinking mode + budget so cells don't clobber each other:
    "<dataset>_<safe_model>_<variant>_think-<on|off>_budget-<ci|final>"
    - <stem>_raw.json     full generation outputs incl. finish_reason / n_tokens
    - <stem>_gens.jsonl   directly consumable by `python -m evaluate.score`
    - <stem>_scored.json  combined metrics + per-problem detailed grading
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .config import BUDGET_TOKENS, EvalConfig, PROMPT_VARIANTS
from .dataset_loader import (
    load_gsm8k,
    load_math500,
    load_math500_stratified,
    load_validation_samples,
)
from .evaluation import compute_extra_metrics
from .generation import run_generation

# Shared course-provided evaluator
from evaluate.score import score_generations


DATASET_LOADERS = {
    "validation":          lambda cfg, limit: load_validation_samples(cfg.repo_root),
    "gsm8k":               lambda cfg, limit: load_gsm8k(limit=limit),
    "math500":             lambda cfg, limit: load_math500(limit=limit),
    # `limit` here means n_per_level (default 20 -> 100 problems balanced over levels 1-5).
    "math500_stratified":  lambda cfg, limit: load_math500_stratified(
        n_per_level=limit if limit else 20
    ),
}


def to_score_format(generation_results: List[Dict]) -> List[Dict]:
    """Convert run_generation() output into the schema evaluate.score expects:
    {prompt, answer, completions: [text, ...]}."""
    return [
        {
            "prompt": r["prompt"],
            "answer": r["answer"],
            "completions": [c["text"] for c in r["completions"]],
        }
        for r in generation_results
    ]


def write_jsonl(items: List[Dict], path: Path) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS))
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of problems (useful for fast iteration).")
    ap.add_argument("--model", default=None,
                    help="Override config.model_id (e.g. a finetuned HF checkpoint).")
    ap.add_argument("--prompt-variant", default="none",
                    choices=list(PROMPT_VARIANTS),
                    help="Local-only user-message prefix; CI ignores this.")
    ap.add_argument("--thinking", default="on", choices=["on", "off"],
                    help="Qwen3 enable_thinking. Local-only; CI uses the "
                         "chat_template default of the pushed checkpoint.")
    ap.add_argument("--budget", default="final", choices=list(BUDGET_TOKENS),
                    help="ci=4096 tokens (matches nightly CI), "
                         "final=16384 tokens (matches final grading). "
                         "Sets max_model_len and max_new_tokens.")
    args = ap.parse_args()

    cfg = EvalConfig(budget=args.budget)
    if args.model:
        cfg.model_id = args.model
    cfg.prompt_variant = args.prompt_variant
    cfg.thinking_enabled = (args.thinking == "on")

    # 1. load
    items = DATASET_LOADERS[args.dataset](cfg, args.limit)
    print(f"[load] {len(items)} problems from {args.dataset}")

    # 2. generate
    print(f"[gen]  model={cfg.model_id}  n={cfg.n_samples}  budget={cfg.budget}"
          f"  max_len={cfg.max_model_len}  variant={cfg.prompt_variant}"
          f"  thinking={args.thinking}")
    generation_results = run_generation(items, cfg)

    # 3. save raw outputs (finish_reason + n_tokens preserved)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = cfg.model_id.replace("/", "_")
    stem = (
        f"{args.dataset}_{safe_model}_{cfg.prompt_variant}"
        f"_think-{args.thinking}_budget-{cfg.budget}"
    )

    raw_path = cfg.results_dir / f"{stem}_raw.json"
    with open(raw_path, "w") as f:
        json.dump(
            {
                "config": {k: str(v) for k, v in asdict(cfg).items()},
                "results": generation_results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"[save] raw    -> {raw_path}")

    # 4. save in the format `evaluate.score` consumes (one row per line)
    score_items = to_score_format(generation_results)
    gens_path = cfg.results_dir / f"{stem}_gens.jsonl"
    write_jsonl(score_items, gens_path)
    print(f"[save] gens   -> {gens_path}")

    # 5. score with the shared evaluator (math -> "boxed" extraction method)
    scored = score_generations(score_items, method="boxed")
    extra = compute_extra_metrics(generation_results)

    # 6. combine and print
    summary = {
        "dataset": args.dataset,
        "model": cfg.model_id,
        "budget": cfg.budget,
        "max_model_len": cfg.max_model_len,
        "prompt_variant": cfg.prompt_variant,
        "thinking": args.thinking,
        "pass@1": scored["metrics"].get("pass@1"),
        "pass@8": scored["metrics"].get("pass@8"),
        "box_compliance_rate": extra["box_compliance_rate"],
        "truncation_rate": extra["truncation_rate"],
        "avg_tokens_per_completion": extra["avg_tokens_per_completion"],
        "n_problems": scored["n_problems"],
        "n_completions_per_problem": scored["n_completions"],
    }
    print("\n=== metrics ===")
    print(json.dumps(summary, indent=2))

    # 7. save full scored output
    scored_path = cfg.results_dir / f"{stem}_scored.json"
    with open(scored_path, "w") as f:
        json.dump(
            {"summary": summary, "scored": scored, "extra": extra},
            f, indent=2, ensure_ascii=False,
        )
    print(f"[save] scored -> {scored_path}")


if __name__ == "__main__":
    main()