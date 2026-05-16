#!/usr/bin/env python3
"""Preprocess raw commonsense_qa JSONL files into evaluation JSONL schema.

Reads `/data/datasets/commonsense_qa/<split>.jsonl` and writes
`/data/preprocessing/commonsense_qa/<split>.jsonl` where each row is:
  {"prompt": "Q...\nA) ...\nB) ...", "answer": "C"}

Usage:
  python scripts/preprocess_commonsense_qa.py --input-dir /data/datasets/commonsense_qa --output-dir /data/preprocessing/commonsense_qa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def choices_to_text(choices) -> List[str]:
    # Handles several possible shapes: list[str], list[dict{text:..}], dict
    if isinstance(choices, list):
        out = []
        for c in choices:
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, dict):
                # common patterns: {'text': '...'} or {'label': '...'}
                if "text" in c:
                    out.append(c["text"])
                elif "label" in c:
                    out.append(c["label"])
                else:
                    # fallback to string representation
                    out.append(str(c))
        return out
    elif isinstance(choices, dict):
        # maybe { 'text': [...]} or other mapping
        for k in ("text", "choices", "options"):
            if k in choices and isinstance(choices[k], list):
                return choices_to_text(choices[k])
        # fallback: convert values
        return [str(v) for v in choices.values()]
    else:
        return [str(choices)]


def answer_key_to_letter(answer_key: Optional[object], n_choices: int) -> Optional[str]:
    if answer_key is None:
        return None
    # If single-letter string (A..Z)
    if isinstance(answer_key, str):
        s = answer_key.strip()
        if len(s) == 1 and s.isalpha():
            return s.upper()
        # maybe it's a numeric string index
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < n_choices:
                return chr(ord("A") + idx)
    if isinstance(answer_key, int):
        idx = int(answer_key)
        if 0 <= idx < n_choices:
            return chr(ord("A") + idx)
    # Unknown form
    return None


def build_prompt(question: str, options: List[str]) -> str:
    parts = [question.strip(), ""]
    for i, opt in enumerate(options):
        parts.append(f"{chr(65+i)}) {opt}")
    return "\n".join(parts)


def process_file(in_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on {in_path}:{lineno}")
                continue

            # Extract question text
            question = None
            for k in ("question", "question__stem", "stem", "query", "text"):
                if k in row:
                    question = row[k]
                    break
            if question is None and "prompt" in row:
                question = row["prompt"]
            if question is None:
                # cannot build prompt
                continue

            # Extract choices
            options = None
            for k in ("choices", "options", "answer_choices", "candidates"):
                if k in row:
                    options = choices_to_text(row[k])
                    break
            # Some variants store choices under 'choices'->'text' or similar
            if options is None and "label" in row and "candidates" in row:
                options = choices_to_text(row.get("candidates"))

            if options is None:
                # fallback: if row has keys like 'A','B','C'
                alpha_opts = [row.get(ch) for ch in ["A", "B", "C", "D", "E"] if ch in row]
                if alpha_opts:
                    options = [str(x) for x in alpha_opts]

            if options is None:
                # cannot process without options
                continue

            # Extract answer key: try common keys
            answer_key = None
            for k in ("answerKey", "answer_key", "label", "answer"):
                if k in row:
                    answer_key = row[k]
                    break

            letter = answer_key_to_letter(answer_key, len(options))
            if letter is None:
                # If dataset uses 'answerIndex' pattern
                if "answerIndex" in row:
                    letter = answer_key_to_letter(row["answerIndex"], len(options))

            if letter is None:
                # No gold answer available; skip (likely test split)
                continue

            prompt = build_prompt(question, options)
            out = {"prompt": prompt, "answer": letter}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {out_path}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/datasets/commonsense_qa"),
        help="Directory with raw jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/preprocessing/commonsense_qa"),
        help="Directory to write preprocessed jsonl files",
    )
    args = parser.parse_args(argv)

    inp = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Input dir not found: {inp}. Run download script first.")

    files = sorted(inp.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No jsonl files found in {inp}")

    total = 0
    for p in files:
        split = p.stem
        out_path = out / f"{split}.jsonl"
        total += process_file(p, out_path)

    print(f"Total written rows: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
