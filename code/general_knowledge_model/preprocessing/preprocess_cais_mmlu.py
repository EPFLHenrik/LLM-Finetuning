#!/usr/bin/env python3
"""Preprocess cais/mmlu test splits into prompt/answer JSONL.

Reads each <config>/test.jsonl under the input directory, formats the
question + choices into a prompt, converts the numeric answer to a letter,
and writes prompt/answer-only rows to the output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def index_to_letter(index: int) -> str:
    return chr(ord("A") + index)


def build_prompt(question: str, choices: list) -> str:
    lines = [question.strip(), ""]
    for i, choice in enumerate(choices):
        lines.append(f"{index_to_letter(i)}. {str(choice).strip()}")
    return "\n".join(lines)


def preprocess_file(input_path: Path, output_path: Path) -> int:
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

            question = row.get("question", "")
            choices = row.get("choices", [])
            answer_index = row.get("answer")

            if not question or not choices or answer_index is None:
                raise SystemExit(f"{input_path}:{lineno}: missing question, choices, or answer")

            prompt = build_prompt(question, choices)
            answer = index_to_letter(int(answer_index))

            fout.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess cais/mmlu test splits.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/datasets/benchmarks/cais__mmlu"),
        help="Directory containing <config>/test.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessing/benchmarks/cais_mmlu"),
        help="Directory to write preprocessed <config>/test.jsonl files",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    configs = sorted(p.parent.name for p in input_dir.glob("*/test.jsonl"))
    if not configs:
        raise SystemExit(f"No test.jsonl files found under {input_dir}")

    total = 0
    for config in configs:
        input_path = input_dir / config / "test.jsonl"
        output_path = output_dir / config / "test.jsonl"
        written = preprocess_file(input_path, output_path)
        print(f"  {config}: {written} rows")
        total += written

    print(f"Done. {len(configs)} subsets, {total} rows total -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
