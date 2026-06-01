"""Multi-dataset weighted evaluation, calibrated against the classroom CI.

Why this exists
---------------
Our long-running MATH-500 pass@8 evaluation saturated at 85-94% for every
SFT/RL checkpoint we shipped, but the classroom CI evaluator reports
base=16-17%, SFT=27-29%, RL=50%. That tells us the classroom set is
(a) pass@1 not pass@8 and (b) weighted toward problems harder than
generic MATH-500. So MATH-500 pass@8 is no longer discriminative for our
checkpoints — it's all noise above 85%.

`eval_v2` reconstructs a difficulty-weighted mixture that is meant to
*correlate* with the classroom CI score, so checkpoint comparisons made
locally actually predict what the official grader will say.

Usage
-----
From the repo root:

    # Calibration run — base model. Target: weighted ~ 14-20%.
    python -m math_task.eval_v2 --model Qwen/Qwen3-1.7B --budget ci

    # Re-evaluate a HF-pushed checkpoint with the 16K final budget:
    python -m math_task.eval_v2 \\
        --model cs-552-2026-mystery-machine/math_model --budget final

    # Quick first-pass on 250 problems total (default profile = "default"),
    # or override with a smaller --profile fast (~120 problems):
    python -m math_task.eval_v2 --model <ckpt> --profile fast

    # Custom subset selection (comma-separated names from SUBSET_REGISTRY):
    python -m math_task.eval_v2 --model <ckpt> --subsets aime_2024,aime_2025,math500_hard

Output
------
math_task/results/eval_v2/<safe_model>_<budget>_<profile>.json
    - per-subset pass@1 + box_compliance + avg_tokens
    - weighted aggregate (matches what the classroom CI should approximate)
    - detailed per-problem grading for failure-mode inspection

Each run also appends one row to math_task/RUNS_V2.md.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import BUDGET_TOKENS, EvalConfig
from .dataset_loader import (
    load_aime_2024,
    load_aime_2025,
    load_gsm8k,
    load_math500_hard,
    load_minerva_math,
    load_olympiad_math,
)
from .evaluation import compute_extra_metrics
from .generation import run_generation

from evaluate.score import score_generations


# ---------------------------------------------------------------- subset registry
# Each subset = name -> (loader, default_n, weight).
#   loader: (n: Optional[int]) -> List[{"prompt", "answer"}]
#   default_n: how many problems to pull in the default profile
#   weight: contribution to the weighted aggregate score
#
# Weights chosen so AIME (very hard, low-base) gets ~2× and Olympiad ~1.5×
# the influence of MATH-500-hard. GSM8K is included at low weight purely as
# a catastrophic-forgetting tripwire — if RL has trashed elementary
# arithmetic, we want the eval to flag it.
SUBSET_REGISTRY: Dict[str, Dict] = {
    "math500_hard": {
        "loader": load_math500_hard,
        "default_n": 80,
        "weight": 1.0,
        "blurb": "MATH-500 level-5 subset (hardest tier).",
    },
    "aime_2024": {
        "loader": load_aime_2024,
        "default_n": 30,
        "weight": 2.0,
        "blurb": "AIME 2024 (30 problems, integer answers).",
    },
    "aime_2025": {
        "loader": load_aime_2025,
        "default_n": 30,
        "weight": 2.0,
        "blurb": "AIME 2025 (30 problems, integer answers).",
    },
    "olympiad_math": {
        "loader": load_olympiad_math,
        "default_n": 60,
        "weight": 1.5,
        "blurb": "OlympiadBench math (EN, open-ended).",
    },
    "minerva_math": {
        "loader": load_minerva_math,
        "default_n": 50,
        "weight": 1.0,
        "blurb": "Hendrycks-MATH level-4/5 (Minerva proxy).",
    },
    "gsm8k_sample": {
        "loader": load_gsm8k,
        "default_n": 50,
        "weight": 0.3,
        "blurb": "GSM8K test sample (forgetting tripwire).",
    },
}


PROFILES = {
    # ~120 problems, ~3 min/ckpt at budget=ci. Use for rapid checkpoint ranking.
    "fast": ["math500_hard", "aime_2024", "aime_2025", "gsm8k_sample"],
    # ~250 problems, ~15 min/ckpt at budget=final. Use for final candidate pick.
    "default": ["math500_hard", "aime_2024", "aime_2025", "olympiad_math", "gsm8k_sample"],
    # Full mixture, ~300 problems. Use for the final 1-2 candidates only.
    "full": list(SUBSET_REGISTRY.keys()),
}


def load_subsets(
    subset_names: List[str],
    n_overrides: Dict[str, int],
) -> List[Dict]:
    """Load and tag all subset items into one flat list.

    Each item is the schema {"prompt", "answer"} plus a "subset" tag we use
    for per-subset scoring downstream.
    """
    out: List[Dict] = []
    for name in subset_names:
        spec = SUBSET_REGISTRY[name]
        n = n_overrides.get(name, spec["default_n"])
        loader: Callable = spec["loader"]
        print(f"[load] {name}: loading up to {n} problems...")
        items = loader(limit=n)
        print(f"[load] {name}: got {len(items)} problems")
        for it in items:
            it["subset"] = name
            out.append(it)
    print(f"[load] total: {len(out)} problems across {len(subset_names)} subsets")
    return out


def score_by_subset(generation_results: List[Dict]) -> Dict[str, Dict]:
    """Split generations by subset tag and run score_generations() per subset."""
    by_subset: Dict[str, List[Dict]] = {}
    for r in generation_results:
        sub = r.get("subset", "_unknown")
        by_subset.setdefault(sub, []).append({
            "prompt": r["prompt"],
            "answer": r["answer"],
            "completions": [c["text"] for c in r["completions"]],
        })

    per_subset: Dict[str, Dict] = {}
    for sub, items in by_subset.items():
        scored = score_generations(items, method="boxed")
        # Also compute extra metrics for this subset.
        sub_raw = [r for r in generation_results if r.get("subset") == sub]
        extras = compute_extra_metrics(sub_raw)
        per_subset[sub] = {
            "n_problems": scored["n_problems"],
            "n_completions": scored["n_completions"],
            "metrics": scored["metrics"],
            "box_compliance_rate": extras["box_compliance_rate"],
            "truncation_rate": extras["truncation_rate"],
            "avg_tokens_per_completion": extras["avg_tokens_per_completion"],
            "detailed_results": scored["detailed_results"],
        }
    return per_subset


def weighted_aggregate(
    per_subset: Dict[str, Dict],
    subset_names: List[str],
) -> Dict:
    """Weighted mean of per-subset pass@1 across `subset_names`.

    Score = sum(w_i * pass@1_i) / sum(w_i)  — unweighted-by-size, so each
    subset's *role* in the eval is controlled by its weight rather than how
    many problems we sampled from it.
    """
    total_w = 0.0
    weighted_pass1 = 0.0
    weighted_pass8 = 0.0
    has_pass8 = False
    for name in subset_names:
        w = SUBSET_REGISTRY[name]["weight"]
        m = per_subset.get(name, {}).get("metrics", {})
        if "pass@1" not in m:
            continue
        total_w += w
        weighted_pass1 += w * m["pass@1"]
        if "pass@8" in m:
            has_pass8 = True
            weighted_pass8 += w * m["pass@8"]

    out = {
        "weighted_pass@1": weighted_pass1 / total_w if total_w else 0.0,
        "subsets_used": list(subset_names),
        "subset_weights": {n: SUBSET_REGISTRY[n]["weight"] for n in subset_names},
    }
    if has_pass8:
        out["weighted_pass@8"] = weighted_pass8 / total_w
    return out


def append_runs_md(
    runs_md: Path,
    *,
    model_id: str,
    profile: str,
    budget: str,
    aggregate: Dict,
    per_subset: Dict,
    when: str,
) -> None:
    header = (
        "| date | model | profile | budget | weighted_pass@1 | "
        + " | ".join([f"{n}_p@1" for n in SUBSET_REGISTRY])
        + " |\n"
    )
    sep = (
        "|---|---|---|---|---|"
        + "|".join(["---"] * len(SUBSET_REGISTRY))
        + "|\n"
    )
    if not runs_md.exists():
        runs_md.write_text("# eval_v2 runs\n\n" + header + sep)

    cells = []
    for name in SUBSET_REGISTRY:
        val = per_subset.get(name, {}).get("metrics", {}).get("pass@1")
        cells.append(f"{val:.4f}" if val is not None else "—")

    row = (
        f"| {when} | {model_id} | {profile} | {budget} "
        f"| {aggregate['weighted_pass@1']:.4f} | "
        + " | ".join(cells)
        + " |\n"
    )
    with open(runs_md, "a") as f:
        f.write(row)


def parse_n_overrides(s: Optional[str]) -> Dict[str, int]:
    """Parse '--n-per-subset math500_hard=20,aime_2024=10' into a dict."""
    if not s:
        return {}
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="HF repo id or local path of the model to eval.")
    ap.add_argument("--profile", default="default", choices=list(PROFILES),
                    help="Subset profile: fast | default | full.")
    ap.add_argument("--subsets", default=None,
                    help="Comma-separated subset names; overrides --profile.")
    ap.add_argument("--n-per-subset", default=None,
                    help="Per-subset n overrides, e.g. "
                         "'math500_hard=40,aime_2024=15'.")
    ap.add_argument("--budget", default="ci", choices=list(BUDGET_TOKENS),
                    help="ci=4096 (fast iteration), final=16384 (final budget). "
                         "Default ci because pass@1 doesn't benefit much from 16K "
                         "vs 4K for ranking purposes.")
    ap.add_argument("--n", type=int, default=1,
                    help="Completions per problem. 1 = pass@1 only (default, "
                         "matches classroom CI semantics). 8 to also get pass@8.")
    ap.add_argument("--thinking", default="on", choices=["on", "off"])
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="0.6 is a common pass@1 sweet spot. Set 0.0 for greedy.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write the result JSON. Default: "
                         "math_task/results/eval_v2/.")
    args = ap.parse_args()

    # ----------------------------------------------------------------- setup
    cfg = EvalConfig(budget=args.budget)
    cfg.model_id = args.model
    cfg.thinking_enabled = (args.thinking == "on")
    cfg.n_samples = args.n
    cfg.temperature = args.temperature

    subset_names = (
        [s.strip() for s in args.subsets.split(",") if s.strip()]
        if args.subsets
        else PROFILES[args.profile]
    )
    unknown = [s for s in subset_names if s not in SUBSET_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown subset(s): {unknown}. "
            f"Choose from {list(SUBSET_REGISTRY)}."
        )

    n_overrides = parse_n_overrides(args.n_per_subset)

    repo_root = cfg.repo_root
    out_dir = args.out_dir or (repo_root / "math_task" / "results" / "eval_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_md = repo_root / "math_task" / "RUNS_V2.md"

    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_model = cfg.model_id.replace("/", "_")
    stem = f"{safe_model}_{args.budget}_{args.profile}_think-{args.thinking}_n-{args.n}"

    print(f"\n=== eval_v2 ===")
    print(f"  model     : {cfg.model_id}")
    print(f"  profile   : {args.profile}  (subsets: {subset_names})")
    print(f"  budget    : {args.budget} ({cfg.max_model_len} tokens)")
    print(f"  n / temp  : {cfg.n_samples} / {cfg.temperature}")
    print(f"  thinking  : {args.thinking}")
    print(f"  out_dir   : {out_dir}")
    print()

    # --------------------------------------------------------------- 1. load
    items = load_subsets(subset_names, n_overrides)

    # ------------------------------------------------------------- 2. gen ALL
    # One vLLM instance handles every subset's prompts in a single batched
    # call — saves the ~30s spin-up cost we'd pay per subset otherwise.
    print(f"\n[gen] launching vLLM for {cfg.model_id}...")
    generation_results = run_generation(items, cfg)

    # Re-attach subset tags lost in run_generation's projection step.
    for gen_r, src in zip(generation_results, items):
        gen_r["subset"] = src["subset"]

    # ----------------------------------------------------------- 3. per-sub
    per_subset = score_by_subset(generation_results)

    # ------------------------------------------------------ 4. weighted agg
    aggregate = weighted_aggregate(per_subset, subset_names)

    # ----------------------------------------------------------- 5. report
    print(f"\n=== per-subset (n_problems · pass@1 · box% · trunc% · avg_tok) ===")
    for name in subset_names:
        s = per_subset.get(name, {})
        if not s:
            print(f"  {name:>16}: (no data)")
            continue
        p1 = s["metrics"].get("pass@1", 0)
        p8 = s["metrics"].get("pass@8")
        line = (
            f"  {name:>16}: {s['n_problems']:>4} · "
            f"p@1={p1:.4f}"
            + (f" p@8={p8:.4f}" if p8 is not None else "")
            + f" · box={s['box_compliance_rate']:.2%}"
            f" · trunc={s['truncation_rate']:.2%}"
            f" · avg_tok={s['avg_tokens_per_completion']:.0f}"
        )
        print(line)

    print(f"\n=== weighted aggregate ===")
    print(f"  weighted_pass@1 = {aggregate['weighted_pass@1']:.4f}")
    if "weighted_pass@8" in aggregate:
        print(f"  weighted_pass@8 = {aggregate['weighted_pass@8']:.4f}")
    print(f"  weights = {aggregate['subset_weights']}")

    # ------------------------------------------------------------- 6. save
    out_path = out_dir / f"{stem}.json"
    summary = {
        "when": when,
        "model": cfg.model_id,
        "profile": args.profile,
        "subsets": subset_names,
        "budget": args.budget,
        "max_model_len": cfg.max_model_len,
        "n_samples": cfg.n_samples,
        "temperature": cfg.temperature,
        "thinking": args.thinking,
        "aggregate": aggregate,
        "per_subset": {
            name: {k: v for k, v in s.items() if k != "detailed_results"}
            for name, s in per_subset.items()
        },
    }
    detailed = {
        "summary": summary,
        "per_subset_detailed": per_subset,  # includes per-problem grading
    }
    out_path.write_text(json.dumps(detailed, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}")

    append_runs_md(
        runs_md, model_id=cfg.model_id, profile=args.profile,
        budget=args.budget, aggregate=aggregate, per_subset=per_subset,
        when=when,
    )
    print(f"[save] appended row to {runs_md}")


if __name__ == "__main__":
    main()
