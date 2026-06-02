#!/usr/bin/env python3
"""Filter a JSONL dataset by a set of categories.

Writes matching rows to an output JSONL file.

Usage examples:
  python scripts/filter_dataset_by_categories.py \
    --input data/preprocessing/merged_distillation.jsonl \
    --output data/datasets/filtered.jsonl

  python scripts/filter_dataset_by_categories.py --help
"""
from __future__ import annotations

import argparse
import json
from typing import Iterable, Set


DEFAULT_CATEGORIES = [
    "professional_law",
    "professional_accounting",
    "college_physics",
    "college_chemistry",
    "college_mathematics",
    "college_biology",
    "high_school_physics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_statistics",
    "high_school_microeconomics",
    "high_school_macroeconomics",
    "high_school_mathematics"
]


def normalize_category(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    # if it's a list or other iterable, pick the first stringy token
    try:
        for v in value:
            if isinstance(v, str):
                return v.strip().lower()
    except Exception:
        pass
    return None


def filter_jsonl(input_path: str, output_path: str, categories: Iterable[str], field: str = "task") -> int:
    categories_set: Set[str] = {c.strip().lower() for c in categories}
    kept = 0
    with open(input_path, "r", encoding="utf-8") as inf, open(output_path, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed lines
                continue

            val = obj.get(field) if isinstance(obj, dict) else None
            # fallback to common keys if the requested field is missing
            if not val and isinstance(obj, dict):
                val = obj.get("task") or obj.get("category")
            norm = normalize_category(val)
            if norm and norm in categories_set:
                outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    return kept


def parse_args():
    p = argparse.ArgumentParser(description="Filter JSONL by category field.")
    p.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    p.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    p.add_argument(
        "--categories",
        "-c",
        help=(
            "Comma-separated list of categories to keep. "
            "Defaults to the predefined set used by the project."
        ),
    )
    p.add_argument("--field", "-f", default="task", help="JSON field containing the category/task (default: 'task'); falls back to 'task' or 'category' if missing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.categories:
        cats = [c for c in (x.strip() for x in args.categories.split(",")) if c]
    else:
        cats = DEFAULT_CATEGORIES

    kept = filter_jsonl(args.input, args.output, cats, field=args.field)
    print(f"Wrote {kept} rows to {args.output}")


if __name__ == "__main__":
    main()
