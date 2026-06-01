#!/usr/bin/env python3
"""Preprocess ScienceQA train.jsonl.

Rules:
- Exclude rows that contain an image (truthy `image` field).
- Create/replace `prompt` with the question plus choices appended as "A) ...\nB) ...".
- Convert the answer to a letter (A, B, C, ...). If answer is an index, map to letter; if it's a string matching a choice, map accordingly.

Writes a JSONL file with the same objects but with `prompt` and `answer` transformed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def index_to_letter(i: int) -> str:
    return chr(ord("A") + i)


def answer_to_letter(answer: Any, choices: list[str]) -> str | None:
    if answer is None:
        return None
    # If numeric index
    try:
        idx = int(answer)
        if 0 <= idx < len(choices):
            return index_to_letter(idx)
    except Exception:
        pass

    # If already a single-letter like 'A' or 'b'
    if isinstance(answer, str) and len(answer.strip()) == 1 and answer.strip().isalpha():
        return answer.strip().upper()

    # If answer is the actual choice text, find it
    if isinstance(answer, str):
        for i, c in enumerate(choices):
            if answer.strip() == str(c).strip():
                return index_to_letter(i)

    return None


def build_prompt(question: str, choices: list[str]) -> str:
    parts = [question.strip()]
    parts.append("")
    for i, c in enumerate(choices):
        parts.append(f"{index_to_letter(i)}) {c}")
    return "\n".join(parts)


def preprocess(input_path: Path, output_path: Path) -> int:
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with input_path.open("r", encoding="utf-8") as fh_in, output_path.open("w", encoding="utf-8") as fh_out:
        for lineno, line in enumerate(fh_in, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skipping invalid JSON on line {lineno}")
                continue

            # Skip rows with an image
            if row.get("image"):
                continue

            question = row.get("question") or row.get("prompt") or ""
            choices = row.get("choices") or []
            if not isinstance(choices, list):
                choices = list(choices)

            prompt = build_prompt(question, choices)

            new_answer = answer_to_letter(row.get("answer"), choices)

            # Update row
            row["prompt"] = prompt
            row["answer"] = new_answer

            fh_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} rows (from {total}) to {output_path}")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ScienceQA train.jsonl")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
