#!/usr/bin/env python3
"""multilingual/train/check_sft_stats.py

Prints size and language/dataset/task distribution of the mixed SFT files.

Usage:
    python -m multilingual.train.check_sft_stats
    python multilingual/train/check_sft_stats.py
"""

import json
from collections import Counter
from pathlib import Path

SFT_DIR = Path(__file__).parents[1] / "data" / "sft" / "mixed"

def check_split(split: str) -> None:
    path = SFT_DIR / f"{split}_mixed.jsonl"
    if not path.exists():
        print(f"[{split}] File not found: {path}")
        return

    records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    n = len(records)
    langs    = Counter(r["language"] for r in records)
    datasets = Counter(r["dataset"]  for r in records)
    tasks    = Counter(r["task_type"] for r in records)

    print(f"=== {split} ({n:,} total) ===")
    print("Languages:")
    for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
        print(f"  {lang:<12} {count:>6}  ({100*count/n:.1f}%)")
    print("Datasets:")
    for ds, count in sorted(datasets.items(), key=lambda x: -x[1]):
        print(f"  {ds:<14} {count:>6}")
    print(f"Tasks:  mcq={tasks['mcq']:,}  math={tasks['math']:,}")
    print()


if __name__ == "__main__":
    for split in ("train", "val"):
        check_split(split)