"""multilingual/eval/select_benchmark_examples.py

Carves out a fixed benchmark test set from the distilled v2_fallback files.
Removes selected examples from all individual _distilled.jsonl files (both
in v2_fallback/ and in v2_fallback/sft_1/) and writes them to a single
combined file: removed_distilled_benchmark_examples.jsonl

Sampling is stratified by (language, subject/domain) per dataset.

Target sources and per-language counts:
  - include:      100 per language (5 languages = 500 total)
  - global_mmlu:  200 per language (5 languages = 1000 total)
  - mera:         100 Russian
  - ceval:        100 Chinese
  - milu:         100 Hindi

Total target: ~1800 examples.

Usage:
    python multilingual/eval/select_benchmark_examples.py \\
        --distill_dir multilingual/data/distilled/v2_fallback \\
        --dry_run
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42

# (dataset_prefix, language, n_to_sample)
BENCHMARK_TARGETS = [
    # INCLUDE — 100 per language
    ("include", "italian",  100),
    ("include", "spanish",  100),
    ("include", "russian",  100),
    ("include", "chinese",  100),
    ("include", "hindi",    100),
    # Global MMLU — 200 per language
    ("global_mmlu", "italian",  300),
    ("global_mmlu", "spanish",  300),
    ("global_mmlu", "russian",  200),
    ("global_mmlu", "chinese",  200),
    ("global_mmlu", "hindi",    200),
    # Language-specific datasets
    ("mera",    "russian", 100),
    ("ceval",   "chinese", 100),
    ("milu",    "hindi",   100),
]


def _strat_key(record: Dict) -> str:
    """Return a stratification key for a record."""
    meta = record.get("metadata") or {}
    # Try subject first, then domain, then regionality
    subject = meta.get("subject") or meta.get("domain") or meta.get("subject_category") or "unknown"
    return str(subject)


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _save_jsonl(records: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _stratified_sample(records: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """Sample n records stratified by subject/domain key."""
    if len(records) <= n:
        return list(records)

    # Group by strat key
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[_strat_key(r)].append(r)

    # Shuffle within each group
    for g in groups.values():
        rng.shuffle(g)

    # Allocate quota proportionally
    total = len(records)
    keys = sorted(groups.keys())
    quota = {k: max(1, round(n * len(groups[k]) / total)) for k in keys}

    # Adjust to hit exactly n
    diff = n - sum(quota.values())
    for k in keys:
        if diff == 0:
            break
        if diff > 0:
            quota[k] += 1
            diff -= 1
        elif quota[k] > 1:
            quota[k] -= 1
            diff += 1

    sampled = []
    for k in keys:
        sampled.extend(groups[k][:quota[k]])

    # Final shuffle
    rng.shuffle(sampled)
    return sampled[:n]


def _find_distilled_files(distill_dir: Path, dataset_prefix: str, language: str) -> List[Path]:
    """Find all *_distilled.jsonl files matching dataset+language in distill_dir."""
    matches = []
    for f in sorted(distill_dir.glob("*_distilled.jsonl")):
        name = f.stem.replace("_distilled", "")
        if name.startswith(dataset_prefix) and language in name:
            matches.append(f)
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill_dir", type=Path,
                        default=Path("multilingual/data/distilled/v2_fallback"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be removed without modifying files")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sft1_dir = args.distill_dir / "sft_1"
    output_path = args.distill_dir / "removed_distilled_benchmark_examples.jsonl"

    if output_path.exists() and not args.dry_run:
        print(f"WARNING: {output_path} already exists — will append")

    all_benchmark: List[Dict] = []
    all_removed_ids: Set[str] = set()

    # ── Step 1: Sample benchmark examples per target ─────────────────────────
    print(f"\n{'='*60}")
    print("SAMPLING BENCHMARK EXAMPLES")
    print(f"{'='*60}")

    for dataset_prefix, language, n_target in BENCHMARK_TARGETS:
        files = _find_distilled_files(args.distill_dir, dataset_prefix, language)
        if not files:
            print(f"  [WARN] No files found for {dataset_prefix} / {language}")
            continue

        # Load all records from matching files
        all_records = []
        for f in files:
            all_records.extend(_load_jsonl(f))

        # Keep only valid distilled examples not already selected
        candidates = [
            r for r in all_records
            if r.get("distill", {}).get("valid", False)
            and (r.get("source_id") or r.get("id", "")) not in all_removed_ids
        ]

        sampled = _stratified_sample(candidates, n_target, rng)
        sampled_ids = {r.get("source_id") or r.get("id", "") for r in sampled}
        all_removed_ids.update(sampled_ids)
        all_benchmark.extend(sampled)

        strat_counts = defaultdict(int)
        for r in sampled:
            strat_counts[_strat_key(r)] += 1

        print(f"  {dataset_prefix:12s} / {language:8s}: {len(sampled):3d} sampled "
              f"from {len(candidates):4d} candidates | "
              f"{len(strat_counts)} strata")

    print(f"\nTotal benchmark examples: {len(all_benchmark)}")
    print(f"Unique source_ids:        {len(all_removed_ids)}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        # Show answer distribution
        from collections import Counter
        lang_counts = Counter(r.get("language") for r in all_benchmark)
        print("\nPer language:", dict(sorted(lang_counts.items())))
        dataset_counts = Counter(r.get("dataset") for r in all_benchmark)
        print("Per dataset: ", dict(sorted(dataset_counts.items())))
        return

    # ── Step 2: Remove from v2_fallback/*_distilled.jsonl ────────────────────
    print(f"\n{'='*60}")
    print("REMOVING FROM v2_fallback/*_distilled.jsonl")
    print(f"{'='*60}")

    total_removed = 0
    for f in sorted(args.distill_dir.glob("*_distilled.jsonl")):
        records = _load_jsonl(f)
        kept = [r for r in records
                if (r.get("source_id") or r.get("id", "")) not in all_removed_ids]
        n_removed = len(records) - len(kept)
        if n_removed > 0:
            _save_jsonl(kept, f)
            print(f"  {f.name}: removed {n_removed}, kept {len(kept)}")
            total_removed += n_removed

    print(f"  Total removed from v2_fallback: {total_removed}")

    # ── Step 3: Remove from sft_1/*_distilled.jsonl ──────────────────────────
    print(f"\n{'='*60}")
    print("REMOVING FROM sft_1/*_distilled.jsonl")
    print(f"{'='*60}")

    total_removed_sft1 = 0
    for f in sorted(sft1_dir.glob("*_distilled.jsonl")):
        records = _load_jsonl(f)
        kept = [r for r in records
                if (r.get("source_id") or r.get("id", "")) not in all_removed_ids]
        n_removed = len(records) - len(kept)
        if n_removed > 0:
            _save_jsonl(kept, f)
            print(f"  {f.name}: removed {n_removed}, kept {len(kept)}")
            total_removed_sft1 += n_removed

    print(f"  Total removed from sft_1: {total_removed_sft1}")

    # ── Step 4: Save benchmark file ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAVING BENCHMARK FILE")
    print(f"{'='*60}")

    with open(output_path, "a", encoding="utf-8") as f:
        for r in all_benchmark:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Written {len(all_benchmark)} records to {output_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    lang_counts = Counter(r.get("language") for r in all_benchmark)
    print("Per language:", dict(sorted(lang_counts.items())))
    dataset_counts = Counter(r.get("dataset") for r in all_benchmark)
    print("Per dataset: ", dict(sorted(dataset_counts.items())))


if __name__ == "__main__":
    main()