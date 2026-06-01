#!/usr/bin/env python3
"""Convert raw MCQ JSONL files into chat-style conversational JSONL.

The output keeps the original metadata, adds a `prompt` field, stores the gold
choice letter in `answer`, and writes a `messages` array with:
- system: the project chat-template system prompt
- user: the formatted MCQ prompt
- assistant: the boxed gold answer, e.g. `\\boxed{A}`

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


def _is_choice_answer(answer: str) -> bool:
    return len(answer) == 1 and answer.isalpha()


def _strip_boxed_answer(answer: str) -> str:
    cleaned = str(answer).strip()
    if cleaned.startswith("\\boxed{") and cleaned.endswith("}"):
        return cleaned[len("\\boxed{") : -1].strip()
    return cleaned


def _assistant_content(answer: str) -> str:
    cleaned = _strip_boxed_answer(answer)
    if _is_choice_answer(cleaned):
        return f"\\boxed{{{cleaned.upper()}}}"
    return cleaned


def _choices_to_options(choices: Any) -> list[tuple[str, str]]:
    if isinstance(choices, dict):
        labels = choices.get("label")
        texts = choices.get("text")
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            options: list[tuple[str, str]] = []
            for label, text in zip(labels, texts):
                label_text = str(label).strip()
                choice_text = str(text).strip()
                if label_text and choice_text:
                    options.append((label_text, choice_text))
            return options
    return []


def _build_mcq_prompt(question: str, options: list[tuple[str, str]]) -> str:
    lines = [question.strip(), ""]
    for label, choice_text in options:
        lines.append(f"{label}) {choice_text}")
    return "\n".join(lines)


def _extract_prompt_and_answer(row: dict[str, Any]) -> tuple[str, str]:
    if isinstance(row.get("messages"), list) and row["messages"]:
        user_content = None
        assistant_content = None
        for message in row["messages"]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role == "user" and content and user_content is None:
                user_content = content
            elif role == "assistant" and content:
                assistant_content = content
        if user_content is not None and assistant_content is not None:
            return user_content, _strip_boxed_answer(assistant_content)

    prompt = None
    if row.get("prompt") is not None:
        prompt_text = str(row["prompt"]).strip()
        if prompt_text:
            prompt = prompt_text

    if prompt is None and row.get("question") is not None and row.get("choices") is not None:
        options = _choices_to_options(row["choices"])
        if options:
            prompt = _build_mcq_prompt(str(row["question"]), options)

    if prompt is None:
        for key in ("question", "input", "text"):
            if row.get(key) is not None:
                prompt_text = str(row[key]).strip()
                if prompt_text:
                    prompt = prompt_text
                    break

    answer = row.get("answer") or row.get("reference") or row.get("label") or row.get("target") or row.get("answerKey")
    if prompt is None or answer is None:
        raise SystemExit(f"Missing prompt or answer in row: {row}")

    return str(prompt).strip(), _strip_boxed_answer(str(answer))


def _normalize_row(row: dict[str, Any], system_message: str) -> dict[str, Any]:
    prompt, answer = _extract_prompt_and_answer(row)
    normalized = dict(row)
    normalized["prompt"] = prompt
    normalized["answer"] = answer
    normalized["messages"] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": _assistant_content(answer)},
    ]
    return normalized


def _convert_file(src: Path, dst: Path, system_message: str) -> int:
    rows = _read_jsonl(src)
    converted = [_normalize_row(row, system_message) for row in rows]
    _write_jsonl(dst, converted)
    return len(converted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MCQ JSONL files into chat-style JSONL.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory containing raw JSONL files")
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