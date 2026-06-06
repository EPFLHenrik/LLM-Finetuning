"""Merge Spain BOE JSONL files into one shuffled JSONL file.

This utility is used for two related BOE artifacts:

1. Raw parsed source articles:
   multilingual/data/raw/spain_boe_legislation/all_spain_boe_legislation.jsonl

2. Generated MCQs:
   multilingual/data/processed/spain_boe_legislation/all_spain_boe_legislation.jsonl

The merge is deterministic: files are read in sorted filename order, duplicate
source_id records are removed, and the final merged list is shuffled with
multilingual.config.VAL_SEED.

Usage for raw parsed articles:
    python -m multilingual.data_pipeline.parsing.merge_boe_jsonl \\
        --input_dir multilingual/data/raw/spain_boe_legislation \\
        --pattern "spain_boe_*.jsonl" \\
        --output_name all_spain_boe_legislation.jsonl \\
        --overwrite

Usage for generated MCQs:
    python -m multilingual.data_pipeline.parsing.merge_boe_jsonl \\
        --input_dir multilingual/data/processed/spain_boe_legislation \\
        --pattern "*_generated.jsonl" \\
        --output_name all_spain_boe_legislation.jsonl \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from multilingual.config import REPO, VAL_SEED


DEFAULT_INPUT_DIR = REPO / "multilingual" / "data" / "processed" / "spain_boe_legislation"
DEFAULT_PATTERN = "*_generated.jsonl"
DEFAULT_OUTPUT_NAME = "all_spain_boe_legislation.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_input_files(input_dir: Path, pattern: str, output_path: Path) -> List[Path]:
    files = sorted(input_dir.glob(pattern))
    return [
        path for path in files
        if path.is_file()
        and path.resolve() != output_path.resolve()
        and path.name != output_path.name
    ]


def merge_jsonl_files(input_dir: Path, pattern: str, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")

    input_files = find_input_files(input_dir, pattern, output_path)
    if not input_files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {input_dir}")

    merged: List[Dict[str, Any]] = []
    seen_source_ids = set()
    duplicate_count = 0

    print(f"Input dir: {input_dir}")
    print(f"Pattern:   {pattern}")
    print(f"Seed:      {VAL_SEED}")

    for path in input_files:
        records = load_jsonl(path)
        kept = 0

        for record in records:
            source_id = str(record.get("source_id", "")).strip()
            if source_id and source_id in seen_source_ids:
                duplicate_count += 1
                continue
            if source_id:
                seen_source_ids.add(source_id)

            metadata = record.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata.setdefault("merged_from_file", path.name)

            merged.append(record)
            kept += 1

        print(f"  {path.name}: read={len(records)}, kept={kept}")

    rng = random.Random(VAL_SEED)
    rng.shuffle(merged)

    write_jsonl(merged, output_path)

    print()
    print(f"Wrote {len(merged)} records -> {output_path}")
    if duplicate_count:
        print(f"Removed {duplicate_count} duplicate source_id records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.input_dir / args.output_name
    merge_jsonl_files(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_path=output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
