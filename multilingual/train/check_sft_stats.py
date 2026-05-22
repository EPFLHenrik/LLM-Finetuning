#!/usr/bin/env python3
"""multilingual/train/check_sft_stats.py

Prints size and language/dataset/task distribution of mixed SFT files.

Usage:
    python -m multilingual.train.check_sft_stats
    python -m multilingual.train.check_sft_stats --sft_dir multilingual/data/sft
    python -m multilingual.train.check_sft_stats --sft_dir multilingual/data/sft/v7

    python multilingual/train/check_sft_stats.py
    python multilingual/train/check_sft_stats.py --sft_dir multilingual/data/sft/v7
"""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_SFT_DIR = Path(__file__).parents[1] / "data" / "sft"


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

    langs = Counter(r["language"] for r in records)
    datasets = Counter(r["dataset"] for r in records)
    tasks = Counter(r["task_type"] for r in records)

    print(f"=== {split} ({n:,} total) ===")

    print("Languages:")
    for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
        print(f"  {lang:<12} {count:>6}  ({100 * count / n:.1f}%)")

    print("Datasets:")
    for ds, count in sorted(datasets.items(), key=lambda x: -x[1]):
        print(f"  {ds:<14} {count:>6}  ({100 * count / n:.1f}%)")

    print("Tasks:")
    for task, count in sorted(tasks.items(), key=lambda x: -x[1]):
        print(f"  {task:<12} {count:>6}  ({100 * count / n:.1f}%)")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft_dir",
        type=Path,
        default=DEFAULT_SFT_DIR,
        help=(
            "Base SFT directory containing the mixed/ folder. "
            "Example: multilingual/data/sft or multilingual/data/sft/v7"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sft_dir = args.sft_dir
    mixed_dir = sft_dir / "mixed"

    print(f"SFT base dir: {sft_dir}")
    print(f"Mixed dir:    {mixed_dir}")
    print()

    for split in ("train", "val"):
        check_split(split, mixed_dir)


if __name__ == "__main__":
    main()