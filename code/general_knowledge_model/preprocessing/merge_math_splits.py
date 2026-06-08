#!/usr/bin/env python3
"""Merge split JSONL files from MATH config subfolders into flat files.

Given a directory like:

    data/datasets/train_datasets/MATH/
        algebra/train.jsonl
        algebra/test.jsonl
        geometry/train.jsonl
        geometry/test.jsonl
        ...

this script writes:

    data/datasets/train_datasets/MATH/train.jsonl
    data/datasets/train_datasets/MATH/test.jsonl

Each output file is the concatenation of the corresponding split files from all
subdirectories, in alphabetical order by config name.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _merge_split(base_dir: Path, split: str) -> int:
    output_path = base_dir / f"{split}.jsonl"
    config_dirs = sorted(path for path in base_dir.iterdir() if path.is_dir())
    total_rows = 0

    with output_path.open("w", encoding="utf-8") as out_handle:
        for config_dir in config_dirs:
            input_path = config_dir / f"{split}.jsonl"
            if not input_path.exists():
                continue
            with input_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if not line.strip():
                        continue
                    out_handle.write(line.rstrip("\n") + "\n")
                    total_rows += 1

    print(f"Wrote {total_rows} rows to {output_path}")
    return total_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge MATH split files from config subfolders")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Directory containing config subfolders like algebra/, geometry/, ...",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    _merge_split(base_dir, "train")
    _merge_split(base_dir, "test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
