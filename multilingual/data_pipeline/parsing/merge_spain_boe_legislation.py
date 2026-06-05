"""Merge all Spain BOE *_generated.jsonl files into one shuffled JSONL file.

The output filename starts with all_ so it appears first alphabetically.

Usage:
    python -m multilingual.data_pipeline.parsing.merge_spain_boe_legislation

or:
    python multilingual.data_pipeline.parsing.merge_spain_boe_legislation.py \
      --input_dir multilingual/data/processed/spain_boe_legislation \
      --output_name all_spain_boe_legislation.jsonl \
      --overwrite
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from multilingual.config import REPO, VAL_SEED

DEFAULT_INPUT_DIR = REPO / "multilingual" / "data" / "processed" / "spain_boe_legislation"
DEFAULT_OUTPUT_NAME = "all_spain_boe_legislation.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_generated_files(input_dir: Path, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")

    generated_files = sorted(input_dir.glob("*_generated.jsonl"))
    generated_files = [p for p in generated_files if p.name != output_path.name]

    if not generated_files:
        raise FileNotFoundError(f"No *_generated.jsonl files found in {input_dir}")

    merged: List[Dict[str, Any]] = []
    seen_source_ids = set()
    duplicate_count = 0

    print(f"Input dir: {input_dir}")
    for path in generated_files:
        records = load_jsonl(path)
        kept = 0
        for record in records:
            source_id = str(record.get("source_id", "")).strip()
            if source_id and source_id in seen_source_ids:
                duplicate_count += 1
                continue
            if source_id:
                seen_source_ids.add(source_id)
            record.setdefault("metadata", {})
            if isinstance(record["metadata"], dict):
                record["metadata"].setdefault("merged_from_file", path.name)
            merged.append(record)
            kept += 1
        print(f"  {path.name}: read={len(records)}, kept={kept}")

    rng = random.Random(VAL_SEED)
    rng.shuffle(merged)

    write_jsonl(merged, output_path)
    print(f"\nWrote {len(merged)} records -> {output_path}")
    if duplicate_count:
        print(f"Removed {duplicate_count} duplicate source_id records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.input_dir / args.output_name
    merge_generated_files(args.input_dir, output_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
