#!/usr/bin/env python3
"""Preprocess COURSEMO/sft_dataset_econ into MCQ prompt/answer JSONL.

Keeps only multiple-choice rows without images and extracts the answer
letter (A-D) from the reasoning field.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ANSWER_PATTERNS = [
    r'(?:the\s+)?(?:correct\s+)?answer\s+(?:is|choice\s+is|choice)\s*[:\-]?\s*\**([A-D])\**',
    r'\b([A-D])\s+is\s+(?:the\s+)?(?:definitely\s+)?correct',
    r'given correct answer of ([A-D])',
    r'answer:\s*([A-D])',
    r'(?:answer|option|choice)\s+([A-D])\b',
]


def _has_image(obj: object) -> bool:
    if isinstance(obj, dict):
        if obj.get("__type__"):
            return True
        return any(_has_image(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_image(v) for v in obj)
    return False


def _extract_answer(row: dict) -> str | None:
    text = row.get("reasoning", "") + " " + row.get("response", "")
    for pat in ANSWER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess sft_dataset_econ to MCQ JSONL.")
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("output", type=Path, help="Output JSONL file")
    args = parser.parse_args(argv)

    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    kept, dropped_image, dropped_no_answer = 0, 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            if not str(row.get("question_type", "")).startswith("Multiple Choice"):
                continue
            if _has_image(row):
                dropped_image += 1
                continue
            answer = _extract_answer(row)
            if answer is None:
                dropped_no_answer += 1
                continue
            out.write(json.dumps({"prompt": row["question"], "answer": answer}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept:               {kept}")
    print(f"Dropped (image):    {dropped_image}")
    print(f"Dropped (no answer):{dropped_no_answer}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
