"""Materialize an evaluation problem mixture to one JSONL and (optionally)
push it to the HuggingFace Hub.

A materialized profile is uploaded to the team dataset under
`benchmarks/<profile>.jsonl` (e.g. `benchmarks/broad.jsonl`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .evaluate import PROFILES, SUBSET_REGISTRY, load_subsets, parse_n_overrides


DEFAULT_REPO = "cs-552-2026-mystery-machine/math"


# Provenance string for each subset — shown in the dataset card so readers
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
# Math Benchmark — {profile} profile

A weighted, multi-source math benchmark for scoring math reasoning models.

## Profile

This file materializes the **`{profile}`** profile (subsets: {subsets_str}).

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
  "weight":  1.0,                  # subset weight in the aggregate
  "source":  "<upstream provenance>"
}}
```

## How to score a model against this

1. Generate `n=8` completions per problem at temperature 0.6, 16384-token budget.
2. Extract the final answer with the standard `\\boxed{{...}}` extractor.
3. Compute pass@k = 1 - C(n - c, k) / C(n, k) per problem, then weighted aggregate.

Minimal scorer:

```python
from datasets import load_dataset
from collections import defaultdict
from code.math_model.evaluation.scoring import score_generations

ds = load_dataset("{repo}", data_files="benchmarks/{profile}.jsonl", split="train")

# After running your model: build {{prompt, answer, completions}} rows per problem.
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

This benchmark is a re-mix of publicly available math evaluation sets. Each
row's `source` field gives the upstream repo. Please cite the original authors
of MATH-500, AIME 2024/2025, OlympiadBench, GSM8K, and Hendrycks et al. (MATH).
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


def push_to_hub(rows: List[Dict], profile: str, repo: str) -> None:
    """Upload the materialized profile to `repo` as `benchmarks/<profile>.jsonl`.

    Requires HF_TOKEN with write scope. Only the JSONL is pushed; the dataset
    card is maintained at the dataset level and is not overwritten here.
    """
    from huggingface_hub import upload_file
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp_path = f.name

    path_in_repo = f"benchmarks/{profile}.jsonl"
    print(f"[hf] uploading {len(rows)} rows to {repo}:{path_in_repo} ...")
    upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"[hf] done → https://huggingface.co/datasets/{repo}/blob/main/{path_in_repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="broad", choices=list(PROFILES),
                    help="Which evaluation profile to materialize (default 'broad').")
    ap.add_argument("--n-per-subset", default=None,
                    help="Override sizes, e.g. 'math500_hard=40,aime_2024=20'.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Local JSONL output path.")
    ap.add_argument("--push-to-hub", nargs="?", const=DEFAULT_REPO, default=None,
                    help="If set, upload to this dataset repo as "
                         f"benchmarks/<profile>.jsonl (default {DEFAULT_REPO}). "
                         "Requires HF_TOKEN with write scope.")
    args = ap.parse_args()

    n_overrides = parse_n_overrides(args.n_per_subset)
    rows = materialize(args.profile, n_overrides)

    write_jsonl(rows, args.out)
    print(f"\n[done] wrote {len(rows)} rows to {args.out}")

    readme = build_readme(args.profile, rows, args.push_to_hub or DEFAULT_REPO)
    readme_path = args.out.with_suffix(".README.md")
    readme_path.write_text(readme)
    print(f"[done] wrote dataset card to {readme_path}")

    if args.push_to_hub:
        push_to_hub(rows, args.profile, args.push_to_hub)


if __name__ == "__main__":
    main()
