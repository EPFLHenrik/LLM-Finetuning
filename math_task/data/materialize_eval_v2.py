"""Materialize the eval_v2 problem mixture to one JSONL and (optionally)
push it to the team HuggingFace Hub as a dataset.

Why this exists
---------------
Teammates (safety, multilingual, general-knowledge experts) need to score
the merged model against the same math benchmark we used locally. Pointing
them at `math_task/eval_v2.py` is fine for runtime, but they shouldn't have
to re-download AIME/MATH-500/OlympiadBench from five different upstream
repos to rebuild our mixture. So we materialize the *exact* problem set
(profile=broad by default) into a single JSONL/parquet, push it to HF, and
they `load_dataset(...)`.

Schema
------
One row per problem:
    {
      "id":      "math500_hard:00",
      "prompt":  "Find ...",
      "answer":  "<gold>",
      "subset":  "math500_hard" | "aime_2024" | ...,
      "weight":  1.0,                       # subset weight in our aggregate
      "source":  "HuggingFaceH4/MATH-500 (level 5)"
    }

Scoring
-------
The schema is consumable directly by `evaluate/score.py` (drop the extra
fields, group completions per prompt, run `score_generations(..., method="boxed")`).
The README we push alongside includes a minimal scoring recipe.

Usage
-----
    # Local-only materialization:
    python -m math_task.data.materialize_eval_v2 \\
        --profile broad \\
        --out /scratch/user/eval_v2_broad.jsonl

    # Materialize + push to HF Hub:
    python -m math_task.data.materialize_eval_v2 \\
        --profile broad \\
        --out /scratch/user/eval_v2_broad.jsonl \\
        --push-to-hub cs-552-2026-mystery-machine/math_eval_v2

    # All available subsets:
    python -m math_task.data.materialize_eval_v2 --profile full --out /scratch/user/eval_v2_full.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from math_task.eval_v2 import PROFILES, SUBSET_REGISTRY, load_subsets, parse_n_overrides


# Provenance string for each subset — shown in the dataset card so teammates
# know exactly where each problem came from.
SOURCES = {
    "math500_broad": "HuggingFaceH4/MATH-500 (stratified 20 per level 1-5)",
    "math500_hard": "HuggingFaceH4/MATH-500 (level 5 only)",
    "aime_2024": "Maxwell-Jia/AIME_2024",
    "aime_2025": "MathArena/aime_2025 (with fallback mirrors)",
    "olympiad_math": "Hothan/OlympiadBench config=OE_TO_maths_en_COMP",
    "minerva_math": "EleutherAI/hendrycks_math (level 4-5)",
    "gsm8k_sample": "openai/gsm8k (test split sample, used as forgetting tripwire)",
}


README_TEMPLATE = """\
---
license: other
language:
- en
task_categories:
- text-generation
tags:
- math
- reasoning
- benchmark
pretty_name: MNLP Mystery Machine — Math Benchmark v2
size_categories:
- n<1K
---

# MNLP Mystery Machine — Math Benchmark v2

A weighted, multi-source math benchmark used by the Mystery Machine team to
score math reasoning models. Built so the four expert tracks
(math / safety / multilingual / general-knowledge) all evaluate the merged
final model against the same fixed problem set.

## Profile

This dataset materializes the **`{profile}`** profile (subsets: {subsets_str}).

| Subset | n | Weight | Source |
|---|---|---|---|
{table_rows}

Total problems: **{total_n}**

## Schema

```python
{{
  "id":      "math500_hard:00",   # unique per row
  "prompt":  "<problem text>",
  "answer":  "<gold answer (Math-Verify compatible)>",
  "subset":  "math500_hard",       # which source bucket this problem is in
  "weight":  1.0,                  # subset weight in the team's aggregate
  "source":  "<upstream provenance>"
}}
```

## How to score a model against this

1. Generate `n=8` completions per problem at temperature 0.6 with the model
   under test. Use a 16384-token budget (final-grade target).
2. Extract the final answer with the standard `\\boxed{{...}}` extractor.
3. Compute pass@k = 1 - C(n - c, k) / C(n, k) per problem, then weighted
   aggregate.

Minimal scorer (assumes you have `evaluate/score.py` from the team repo):

```python
from datasets import load_dataset
from collections import defaultdict
from evaluate.score import score_generations

ds = load_dataset("{repo}", split="test")

# After running your model: build {{prompt, answer, completions}} rows per problem.
# Group by subset, score each, then aggregate by weight.
by_subset = defaultdict(list)
for row, completions in zip(ds, all_completions):
    by_subset[row["subset"]].append({{
        "prompt": row["prompt"], "answer": row["answer"], "completions": completions
    }})

weighted, total_w = 0.0, 0.0
for sub, items in by_subset.items():
    scored = score_generations(items, method="boxed")
    w = items[0].get("weight", 1.0)  # all rows in a subset share the same weight
    p8 = scored["metrics"]["pass@8"]
    weighted += w * p8
    total_w += w
print(f"weighted_pass@8 = {{weighted/total_w:.4f}}")
```

## Citation / provenance

This benchmark is a *re-mix* of publicly available math evaluation sets.
Each row's `source` field gives the upstream repo. Please cite the original
authors of MATH-500, AIME 2024/2025, OlympiadBench, GSM8K, and Hendrycks et
al. (MATH) when using this benchmark.
"""


def materialize(profile: str, n_overrides: Dict[str, int]) -> List[Dict]:
    """Load every subset in `profile`, return a flat list of rows."""
    subset_names = PROFILES[profile]
    items = load_subsets(subset_names, n_overrides)

    rows: List[Dict] = []
    # Per-subset running indices so each id is "<subset>:NN"
    counters: Dict[str, int] = {}
    for it in items:
        sub = it["subset"]
        i = counters.get(sub, 0)
        counters[sub] = i + 1
        rows.append({
            "id": f"{sub}:{i:03d}",
            "prompt": it["prompt"],
            "answer": str(it["answer"]),
            "subset": sub,
            "weight": SUBSET_REGISTRY[sub]["weight"],
            "source": SOURCES.get(sub, "unknown"),
        })
    return rows


def write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_readme(profile: str, rows: List[Dict], repo: str) -> str:
    by_subset: Dict[str, List[Dict]] = {}
    for r in rows:
        by_subset.setdefault(r["subset"], []).append(r)

    table_lines = []
    for sub in PROFILES[profile]:
        n = len(by_subset.get(sub, []))
        w = SUBSET_REGISTRY[sub]["weight"]
        src = SOURCES.get(sub, "unknown")
        table_lines.append(f"| `{sub}` | {n} | {w} | {src} |")

    return README_TEMPLATE.format(
        profile=profile,
        subsets_str=", ".join(f"`{s}`" for s in PROFILES[profile]),
        table_rows="\n".join(table_lines),
        total_n=len(rows),
        repo=repo,
    )


def push_to_hub(rows: List[Dict], readme: str, repo: str) -> None:
    """Push as a HF dataset under `repo`. Requires HF_TOKEN with write scope."""
    from datasets import Dataset
    from huggingface_hub import HfApi, upload_file
    import tempfile

    ds = Dataset.from_list(rows)
    print(f"[hf] pushing {len(rows)} rows to {repo} (split=test)...")
    ds.push_to_hub(repo, split="test", private=False)

    # Push README separately so it shows up as the dataset card.
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(readme)
        readme_path = f.name
    print(f"[hf] pushing README.md to {repo}...")
    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"[hf] done → https://huggingface.co/datasets/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="broad", choices=list(PROFILES),
                    help="Which eval_v2 profile to materialize. "
                         "Default 'broad' (the calibrated one we use locally).")
    ap.add_argument("--n-per-subset", default=None,
                    help="Override sizes, e.g. 'math500_hard=40,aime_2024=20'.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Local JSONL output path.")
    ap.add_argument("--push-to-hub", default=None,
                    help="If set, push as a HF dataset under this repo "
                         "(e.g. cs-552-2026-mystery-machine/math_eval_v2). "
                         "Requires HF_TOKEN with write scope.")
    args = ap.parse_args()

    n_overrides = parse_n_overrides(args.n_per_subset)
    rows = materialize(args.profile, n_overrides)

    write_jsonl(rows, args.out)
    print(f"\n[done] wrote {len(rows)} rows to {args.out}")

    readme = build_readme(args.profile, rows,
                          args.push_to_hub or "<your-repo>")
    readme_path = args.out.with_suffix(".README.md")
    readme_path.write_text(readme)
    print(f"[done] wrote dataset card to {readme_path}")

    if args.push_to_hub:
        push_to_hub(rows, readme, args.push_to_hub)


if __name__ == "__main__":
    main()
