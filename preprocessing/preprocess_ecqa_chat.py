#!/usr/bin/env python3
"""Convert ECQA JSONL files into chat-style conversational JSONL.

The output keeps the original metadata, adds a `prompt` field, stores the gold
choice letter in `answer`, and writes a `messages` array with:
- system: the project chat-template system prompt
- user: the ECQA question followed by lettered options
- assistant: the requested starter sentence, `taskB`, and the boxed gold
    answer, e.g. `\\boxed{A}`

This matches the conversational shape expected by TRL when
`assistant_only_loss=True` is enabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_MESSAGE = (
    "You are a careful reasoning assistant.\n\n"
    "Reason through the problem when needed.\n\n"
    "Your final answer must appear exactly once inside \\boxed{...}.\n\n"
    "Rules:\n"
    "- For multiple-choice questions, output only the correct capital letter inside \\boxed{...}.\n"
    "- For math questions, output only the final answer inside \\boxed{...}.\n"
    "- Do not include explanations or extra text inside the box.\n"
    "- Never output multiple boxed answers.\n"
    "- Never leave the final answer unboxed.\n\n"
    "Examples:\n"
    "\\boxed{B}\n"
    "\\boxed{42}"
)

CHOICE_LABELS = ("A", "B", "C", "D", "E")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{lineno}: each row must be a JSON object")
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_boxed_answer(answer: str) -> str:
    cleaned = str(answer).strip()
    if cleaned.startswith("\\boxed{") and cleaned.endswith("}"):
        return cleaned[len("\\boxed{") : -1].strip()
    return cleaned


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _build_options(row: dict[str, Any]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for index, label in enumerate(CHOICE_LABELS, start=1):
        option_key = f"q_op{index}"
        if option_key not in row or row[option_key] is None:
            raise SystemExit(f"Missing {option_key} in row: {row}")
        option_text = str(row[option_key]).strip()
        if not option_text:
            raise SystemExit(f"Empty {option_key} in row: {row}")
        options.append((label, option_text))
    return options


def _build_prompt(question: str, options: list[tuple[str, str]]) -> str:
    lines = [str(question).strip(), ""]
    for label, option_text in options:
        lines.append(f"{label}){option_text}")
    return "\n".join(lines)


def _answer_letter(options: list[tuple[str, str]], answer_text: Any) -> str:
    target = _normalize_text(_strip_boxed_answer(str(answer_text)))
    if not target:
        raise SystemExit("Missing q_ans value")

    for label, option_text in options:
        if _normalize_text(option_text) == target:
            return label

    lowered_target = target.casefold()
    for label, option_text in options:
        if _normalize_text(option_text).casefold() == lowered_target:
            return label

    raise SystemExit(f"q_ans does not match any ECQA option: {target!r}")


def _assistant_content(row: dict[str, Any], answer: str) -> str:
    if row.get("taskB") is None:
        raise SystemExit(f"Missing taskB in row: {row}")

    task_b = str(row["taskB"]).strip()
    if not task_b:
        raise SystemExit(f"Empty taskB in row: {row}")

    think_block = f"<think>\nLets think about this step by step\n\n{task_b}\n</think>"
    return "\n\n".join([
        think_block,
        f"So the answer is: \\boxed{{{answer}}}",
    ])


def _normalize_row(row: dict[str, Any], system_message: str) -> dict[str, Any]:
    if row.get("q_text") is None:
        raise SystemExit(f"Missing q_text in row: {row}")
    if row.get("q_ans") is None:
        raise SystemExit(f"Missing q_ans in row: {row}")

    options = _build_options(row)
    prompt = _build_prompt(str(row["q_text"]), options)
    answer = _answer_letter(options, row["q_ans"])
    assistant_content = _assistant_content(row, answer)

    normalized = dict(row)
    normalized["prompt"] = prompt
    normalized["answer"] = answer
    normalized["messages"] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    return normalized


def _convert_file(src: Path, dst: Path, system_message: str) -> int:
    rows = _read_jsonl(src)
    converted = [_normalize_row(row, system_message) for row in rows]
    _write_jsonl(dst, converted)
    return len(converted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ECQA JSONL files into chat-style JSONL.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory containing raw ECQA JSONL files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write chat-formatted JSONL files")
    parser.add_argument(
        "--system-message",
        type=str,
        default=SYSTEM_MESSAGE,
        help="System prompt to insert as the first chat message",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input dir not found: {args.input_dir}")
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output dir already exists: {args.output_dir} (pass --overwrite)")

    total = 0
    files = sorted(args.input_dir.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"No jsonl files found under {args.input_dir}")

    for src in files:
        dst = args.output_dir / src.relative_to(args.input_dir)
        total += _convert_file(src, dst, args.system_message)
        print(f"Wrote {dst}")

    print(f"Converted {len(files)} files / {total} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())