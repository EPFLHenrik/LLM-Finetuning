#!/usr/bin/env python3
"""Preprocess raw JSONL files from a Hugging Face dataset into eval schema.

Usage:
    python scripts/preprocess_hf.py --dataset-path <dataset> [--dataset-config <config>] --input-dir <dir> --output-dir <dir> --question-keys <keys> --choice-keys <keys> --answer-keys <keys>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional


def dataset_slug(dataset_path: str, dataset_config: str | None) -> str:
    slug = dataset_path.replace("/", "__").replace(":", "__")
    if dataset_config:
        slug += f"__{dataset_config.replace('/', '__').replace(':', '__').replace(' ', '_')}"
    return slug
def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def choices_to_text(choices) -> List[str]:
    if isinstance(choices, list):
        out = []
        for c in choices:
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, dict):
                if "text" in c:
                    out.append(c["text"])
                elif "label" in c:
                    out.append(c["label"])
                else:
                    out.append(str(c))
        return out
    if isinstance(choices, dict):
        for k in ("text", "choices", "options"):
            if k in choices and isinstance(choices[k], list):
                return choices_to_text(choices[k])
        return [str(v) for v in choices.values()]
    return [str(choices)]


def answer_key_to_letter(answer_key: Optional[object], n_choices: int) -> Optional[str]:
    if answer_key is None:
        return None
    if isinstance(answer_key, str):
        s = answer_key.strip()
        if len(s) == 1 and s.isalpha():
            return s.upper()
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < n_choices:
                return chr(ord("A") + idx)
    if isinstance(answer_key, int):
        idx = int(answer_key)
        if 0 <= idx < n_choices:
            return chr(ord("A") + idx)
    return None


def build_prompt(question: str, options: List[str]) -> str:
    parts = [question.strip(), ""]
    for i, opt in enumerate(options):
        parts.append(f"{chr(65 + i)}) {opt}")
    return "\n".join(parts)


def find_first_value(row: dict, keys: Iterable[str]) -> object | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def process_file(
    in_path: Path,
    out_path: Path,
    question_keys: List[str],
    choice_keys: List[str],
    answer_keys: List[str],
) -> int:
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

            question = find_first_value(row, question_keys)
            if question is None:
                continue

            options = None
            for key in choice_keys:
                if key in row:
                    options = choices_to_text(row[key])
                    break

            if options is None and "label" in row and "candidates" in row:
                options = choices_to_text(row.get("candidates"))

            if options is None:
                alpha_opts = [row.get(ch) for ch in [chr(65 + i) for i in range(26)] if ch in row]
                if alpha_opts:
                    options = [str(x) for x in alpha_opts]

            if options is None:
                continue

            answer_key = find_first_value(row, answer_keys)
            letter = answer_key_to_letter(answer_key, len(options))
            if letter is None and "answerIndex" in row:
                letter = answer_key_to_letter(row["answerIndex"], len(options))
            if letter is None:
                continue

            prompt = build_prompt(str(question), options)
            fout.write(json.dumps({"prompt": prompt, "answer": letter}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {out_path}")
    return written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Hugging Face dataset path, for example tau/commonsense_qa",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional Hugging Face dataset config/subset, for example ARC-Challenge",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the raw JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write preprocessed JSONL files",
    )
    parser.add_argument(
        "--question-keys",
        required=True,
        help="Comma-separated keys used to find the question text",
    )
    parser.add_argument(
        "--choice-keys",
        required=True,
        help="Comma-separated keys used to find the answer choices",
    )
    parser.add_argument(
        "--answer-keys",
        required=True,
        help="Comma-separated keys used to find the gold answer",
    )
    args = parser.parse_args(argv)

    _ = dataset_slug(args.dataset_path, args.dataset_config)

    inp = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Input dir not found: {inp}. Run download script first.")

    files = sorted(inp.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No jsonl files found in {inp}")

    question_keys = parse_csv_list(args.question_keys)
    choice_keys = parse_csv_list(args.choice_keys)
    answer_keys = parse_csv_list(args.answer_keys)

    total = 0
    for p in files:
        total += process_file(p, out / f"{p.stem}.jsonl", question_keys, choice_keys, answer_keys)

    print(f"Total written rows: {total}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))