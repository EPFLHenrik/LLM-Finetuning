#!/usr/bin/env python3
"""Preprocess MMLU auxiliary_train JSONL into prompt/answer JSONL.

Input rows may be wrapped as {"train": {...}} as produced by scripts/download_hf.py.
Output rows are normalized to:
  {
    "prompt": "<question>\nChoices:\nA) ...\nB) ...",
    "answer": "A"
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _unwrap_row(row: dict[str, Any]) -> dict[str, Any]:
    # download_hf.py can wrap each row with the split name, e.g. {"train": {...}}
    if {"train", "validation", "test"} & set(row.keys()):
        for key in ("train", "validation", "test"):
            value = row.get(key)
            if isinstance(value, dict):
                return value
    return row


def _answer_to_letter(answer: Any, n_choices: int) -> str:
    if isinstance(answer, int):
        index = answer
    elif isinstance(answer, str):
        cleaned = answer.strip()
        if cleaned.isdigit():
            index = int(cleaned)
        elif len(cleaned) == 1 and cleaned.upper() in LETTERS:
            return cleaned.upper()
        else:
            raise ValueError(f"Unsupported answer string: {answer!r}")
    else:
        raise ValueError(f"Unsupported answer type: {type(answer).__name__}")

    if index < 0 or index >= n_choices:
        raise ValueError(f"Answer index out of range: {index} for {n_choices} choices")
    if index >= len(LETTERS):
        raise ValueError(f"Too many choices ({n_choices}); only up to {len(LETTERS)} supported")
    return LETTERS[index]


def _build_prompt(question: str, choices: list[Any]) -> str:
    parts = [question.rstrip(), "Choices:"]
    for idx, choice in enumerate(choices):
        label = LETTERS[idx]
        parts.append(f"{label}) {str(choice).strip()}")
    return "\n".join(parts)


def preprocess_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    total = 0
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                continue
            total += 1

            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{input_path}:{lineno}: invalid JSON: {exc}") from exc

            if not isinstance(raw, dict):
                raise SystemExit(f"{input_path}:{lineno}: expected JSON object per line")

            row = _unwrap_row(raw)
            question = str(row.get("question", "")).strip()
            choices = row.get("choices")
            answer = row.get("answer")

            if not question:
                raise SystemExit(f"{input_path}:{lineno}: missing question")
            if not isinstance(choices, list) or not choices:
                raise SystemExit(f"{input_path}:{lineno}: missing or invalid choices list")
            if len(choices) > len(LETTERS):
                raise SystemExit(f"{input_path}:{lineno}: too many choices ({len(choices)})")

            try:
                answer_letter = _answer_to_letter(answer, len(choices))
            except ValueError as exc:
                raise SystemExit(f"{input_path}:{lineno}: {exc}") from exc

            out_row = {
                "prompt": _build_prompt(question, choices),
                "answer": answer_letter,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    return total, written


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess cais/mmlu auxiliary_train JSONL")
    parser.add_argument("input", type=Path, help="Path to input JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <input>.preprocessed.jsonl)",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}.preprocessed.jsonl")
    )

    total, written = preprocess_file(input_path, output_path)
    print(f"Read {total} row(s)")
    print(f"Wrote {written} row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
