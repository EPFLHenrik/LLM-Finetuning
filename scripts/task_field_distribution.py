#!/usr/bin/env python3
"""Print the distribution of a field in a JSONL/JSON dataset.

This is useful for checking how many examples each `task` value has in a
downloaded dataset before preprocessing or training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{lineno}: each row must be a JSON object")
                rows.append(row)
        return rows

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            return list(payload)
        raise SystemExit(f"{path}: JSON files must contain a list of objects")

    raise SystemExit(f"Unsupported dataset format: {path.suffix}. Use .jsonl or .json")


def _normalize_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, str):
        text = value.strip()
        return text if text else "<empty>"
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the distribution of a field in a JSONL/JSON dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Input .jsonl or .json dataset file")
    parser.add_argument("--field", default="task", help="Field to count (default: task)")
    parser.add_argument("--top-k", type=int, default=0, help="Only print the top K values (0 = print all)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = _load_rows(args.input)

    counts = Counter()
    for row in rows:
        counts[_normalize_value(row.get(args.field))] += 1

    total = sum(counts.values())
    print(f"Field: {args.field}")
    print(f"Rows: {total}")
    print("Distribution:")

    items = counts.most_common(args.top_k or None)
    for value, count in items:
        pct = (count / total * 100.0) if total else 0.0
        print(f"{value}\t{count}\t{pct:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())