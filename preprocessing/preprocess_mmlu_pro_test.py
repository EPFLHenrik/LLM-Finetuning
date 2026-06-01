#!/usr/bin/env python3
"""Preprocess the MMLU-Pro test JSONL into prompt/answer format.

The input rows are expected to contain at least:
- `question`: the question stem
- `options`: a list of answer choices
- `answer`: the gold answer letter or answer index

This script rewrites each row so that:
- `question` is renamed to `prompt`
- the prompt appends the options as:

  Question stem

  A) option 1
  B) option 2
  ...

- the answer is normalized to a capital letter when possible
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def index_to_letter(index: int) -> str:
    if index < 0:
        raise ValueError(f"negative option index: {index}")
    if index >= 26:
        raise ValueError(f"option index out of supported range: {index}")
    return chr(ord("A") + index)


def build_prompt(question: str, options: list[Any]) -> str:
    parts = [question.strip(), ""]
    for index, option in enumerate(options):
        parts.append(f"{index_to_letter(index)}) {str(option).strip()}")
    return "\n".join(parts)


def normalize_answer(answer: Any, options: list[Any], answer_index: Any = None) -> str | None:
    if isinstance(answer, str):
        cleaned = answer.strip()
        if len(cleaned) == 1 and cleaned.isalpha():
            return cleaned.upper()
        if cleaned.isdigit():
            index = int(cleaned)
            if 0 <= index < len(options):
                return index_to_letter(index)

    if isinstance(answer, int) and 0 <= answer < len(options):
        return index_to_letter(answer)

    if answer_index is not None:
        try:
            index = int(answer_index)
        except (TypeError, ValueError):
            index = None
        if index is not None and 0 <= index < len(options):
            return index_to_letter(index)

    return str(answer).strip().upper() if answer is not None else None


def preprocess(input_path: Path, output_path: Path) -> int:
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{input_path}:{lineno}: invalid JSON: {exc}") from exc

            if not isinstance(row, dict):
                raise SystemExit(f"{input_path}:{lineno}: each row must be a JSON object")

            question = row.get("question")
            options = row.get("options")

            if not isinstance(question, str) or not question.strip():
                raise SystemExit(f"{input_path}:{lineno}: missing question text")
            if not isinstance(options, list) or not options:
                raise SystemExit(f"{input_path}:{lineno}: missing options list")

            prompt = build_prompt(question, options)
            normalized_answer = normalize_answer(row.get("answer"), options, row.get("answer_index"))

            updated_row = dict(row)
            updated_row["prompt"] = prompt
            updated_row.pop("question", None)
            if normalized_answer is not None:
                updated_row["answer"] = normalized_answer

            fout.write(json.dumps(updated_row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {output_path}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess the MMLU-Pro test JSONL into prompt format.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/datasets/benchmarks/MMLU-pro/test.jsonl"),
        help="Path to the raw MMLU-Pro test JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/benchmarks/MMLU-pro/preprocessed/test.jsonl"),
        help="Path to write the preprocessed JSONL",
    )
    args = parser.parse_args()
    preprocess(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())