#!/usr/bin/env python3
"""Convert swissai_merged.jsonl flat format to chat messages format for lora_post_train.py.

Input fields per row: prompt, answer, completion, system_message (+ others ignored).
Output adds a `messages` list: [system, user, assistant] where the assistant content
wraps the reasoning in <think>...</think> and ends with \boxed{answer}.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BOXED_RE = re.compile(r"(.*?)(\s*\\boxed\{[^}]*\}\s*)$", re.DOTALL)


def build_assistant_content(completion: str, answer: str) -> str:
    m = BOXED_RE.match(completion)
    if m:
        reasoning = m.group(1).strip()
        boxed = m.group(2).strip()
    else:
        reasoning = completion.strip()
        boxed = f"\\boxed{{{answer}}}"
    return f"<think>\n{reasoning}\n</think>\n\n{boxed}"


def convert(input_path: Path, output_path: Path) -> None:
    rows_in = 0
    rows_out = 0
    skipped = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows_in += 1
            row = json.loads(line)

            prompt = row.get("prompt", "")
            answer = row.get("answer", "")
            completion = row.get("completion", "")
            system_message = row.get("system_message", "")

            if not prompt or not completion:
                skipped += 1
                continue

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": build_assistant_content(completion, answer)},
            ]

            out = {"prompt": prompt, "answer": answer, "messages": messages}
            fout.write(json.dumps(out) + "\n")
            rows_out += 1

    print(f"Read {rows_in} rows, wrote {rows_out}, skipped {skipped}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert swissai flat JSONL to messages format.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/preprocessing/swissai_merged.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/preprocessing/swissai_merged.jsonl",
        help="Output path (defaults to overwriting input in-place).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
