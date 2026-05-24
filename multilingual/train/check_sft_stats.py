#!/usr/bin/env python3
"""multilingual/train/check_sft_stats.py

Prints size and language/dataset/task distribution of mixed SFT files.
Shows distilled (D) / raw (R) / total (T) counts per dataset and language.

Usage:
    python -m multilingual.train.check_sft_stats
    python -m multilingual.train.check_sft_stats --sft_dir multilingual/data/sft/v10
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_SFT_DIR = Path(__file__).parents[1] / "data" / "sft"


def _drt(d: int, r: int) -> str:
    """Format as e.g. '25D / 30R / 55T'"""
    return f"{d}D / {r}R / {d+r}T"


def check_split(split: str, mixed_dir: Path) -> None:
    path = mixed_dir / f"{split}_mixed.jsonl"
    if not path.exists():
        print(f"[{split}] File not found: {path}")
        return

    records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    n = len(records)

    if n == 0:
        print(f"=== {split} (0 total) ===")
        print("Empty file.")
        print()
        return

    distilled = [r for r in records if r.get("metadata", {}).get("distilled")]
    raw       = [r for r in records if not r.get("metadata", {}).get("distilled")]

    langs    = Counter(r["language"] for r in records)
    datasets = Counter(r["dataset"]  for r in records)
    tasks    = Counter(r["task_type"] for r in records)

    ds_dist   = Counter(r["dataset"]  for r in distilled)
    ds_raw    = Counter(r["dataset"]  for r in raw)
    lang_dist = Counter(r["language"] for r in distilled)
    lang_raw  = Counter(r["language"] for r in raw)

    print(f"=== {split} ({n:,} total | {len(distilled):,}D / {len(raw):,}R) ===")

    print("\nLanguages:")
    for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
        d = lang_dist.get(lang, 0)
        r = lang_raw.get(lang, 0)
        print(f"  {lang:<14} {count:>6}  ({100*count/n:.1f}%)   {_drt(d, r)}")

    print("\nDatasets:")
    for ds, count in sorted(datasets.items(), key=lambda x: -x[1]):
        d = ds_dist.get(ds, 0)
        r = ds_raw.get(ds, 0)
        print(f"  {ds:<18} {count:>6}  ({100*count/n:.1f}%)   {_drt(d, r)}")

    print("\nTasks:")
    for task, count in sorted(tasks.items(), key=lambda x: -x[1]):
        print(f"  {task:<12} {count:>6}  ({100*count/n:.1f}%)")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft_dir", type=Path, default=DEFAULT_SFT_DIR,
        help="Base SFT directory containing the mixed/ folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mixed_dir = args.sft_dir / "mixed"
    print(f"SFT dir:   {args.sft_dir}")
    print(f"Mixed dir: {mixed_dir}")
    print()
    for split in ("train", "val"):
        check_split(split, mixed_dir)


if __name__ == "__main__":
    main()