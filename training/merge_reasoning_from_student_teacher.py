#!/usr/bin/env python3
"""Merge student and teacher outputs into a canonical SFT JSONL file.

The script reads two JSONL files with matching prompts and answers. If the
student row is marked correct, the student source is kept and its final boxed
answer is normalized to ``\boxed{<gold>}`` when the model used a fallback boxed
format such as ``\boxed{\text{A}}``. If the student row is not correct, the
teacher source is used unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.extract_answer import extract_boxed_answer_with_fallbacks
from evaluate.extract_answer import last_boxed_only_string


PROMPT_KEYS = ("prompt", "question", "input", "text")
ANSWER_KEYS = ("gold_answer", "answer", "reference", "label", "target", "answerKey", "boxed_answer")
REASONING_KEYS = ("actual_answer", "completion", "reasoning", "response", "output", "text")
REASON_KEYS = ("reason", "strategy", "extraction_strategy")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{lineno}: each row must be a JSON object")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: file is empty")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _get_first(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "0", "no", "n"}:
            return False
    return None


def _build_boxed_answer(answer: str) -> str:
    return f"\\boxed{{{answer.strip()}}}"


def _normalize_boxed_answer(text: str, answer: str) -> tuple[str, str]:
    extracted, strategy = extract_boxed_answer_with_fallbacks(text)
    if extracted is None:
        return text, strategy

    gold = answer.strip().upper()
    if extracted.upper() != gold:
        return text, strategy

    if strategy == "boxed_exact_capital":
        return text, strategy

    boxed = last_boxed_only_string(text)
    if boxed is not None:
        normalized = text[: text.rfind(boxed)] + _build_boxed_answer(gold) + text[text.rfind(boxed) + len(boxed) :]
    else:
        stripped = text.rstrip()
        if stripped:
            normalized = stripped + "\n\n" + _build_boxed_answer(gold)
        else:
            normalized = _build_boxed_answer(gold)
    return normalized, strategy


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    prompt = _get_first(row, PROMPT_KEYS)
    if prompt is None:
        raise SystemExit(f"Missing prompt-like field in row: {row}")
    answer = _get_first(row, ANSWER_KEYS) or ""
    return prompt, answer


def _load_indexed_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_jsonl(path)
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        key = _row_key(row)
        if key in indexed:
            raise SystemExit(f"{path}:{index}: duplicate row key for prompt/answer pair")
        indexed[key] = row
    return indexed


def _select_reasoning(student_row: dict[str, Any], teacher_row: dict[str, Any]) -> tuple[str, str, bool, str]:
    student_correct = _as_bool(student_row.get("correct"))
    if student_correct is None:
        raise SystemExit(f"Missing boolean 'correct' field in student row: {student_row}")

    student_reasoning = _get_first(student_row, REASONING_KEYS)
    teacher_reasoning = _get_first(teacher_row, REASONING_KEYS)
    student_strategy = _get_first(student_row, REASON_KEYS) or ""

    if student_reasoning is None:
        raise SystemExit(f"Missing reasoning field in student row: {student_row}")
    if teacher_reasoning is None:
        raise SystemExit(f"Missing reasoning field in teacher row: {teacher_row}")

    if student_correct:
        return student_reasoning, "student", True, student_strategy
    return teacher_reasoning, "teacher", False, student_strategy


def _build_messages(prompt: str, reasoning: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reasoning},
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge student and teacher reasoning JSONL files into one SFT-ready JSONL file."
    )
    parser.add_argument("student", type=Path, help="Student JSONL file with correctness and reasoning")
    parser.add_argument("teacher", type=Path, help="Teacher JSONL file with reasoning")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <student>.merged_reasoning.jsonl)",
    )
    args = parser.parse_args(argv)

    student_path = args.student.expanduser().resolve()
    teacher_path = args.teacher.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else student_path.with_suffix(".merged_reasoning.jsonl")

    if not student_path.exists():
        raise FileNotFoundError(f"Student file not found: {student_path}")
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher file not found: {teacher_path}")

    student_rows = _load_indexed_rows(student_path)
    teacher_rows = _load_indexed_rows(teacher_path)

    missing_teacher = sorted(set(student_rows) - set(teacher_rows))
    missing_student = sorted(set(teacher_rows) - set(student_rows))
    if missing_teacher or missing_student:
        problems: list[str] = []
        if missing_teacher:
            problems.append(f"missing {len(missing_teacher)} teacher row(s)")
        if missing_student:
            problems.append(f"missing {len(missing_student)} student row(s)")
        raise SystemExit("Key mismatch between files: " + ", ".join(problems))

    merged_rows: list[dict[str, Any]] = []
    student_correct_count = 0

    for key, student_row in student_rows.items():
        teacher_row = teacher_rows[key]
        selected_reasoning, source, student_correct, student_strategy = _select_reasoning(student_row, teacher_row)
        student_correct_count += int(student_correct)

        prompt = _get_first(student_row, PROMPT_KEYS)
        assert prompt is not None
        answer = _get_first(student_row, ANSWER_KEYS) or _get_first(teacher_row, ANSWER_KEYS)
        if answer is None:
            raise SystemExit(f"Missing answer field for row: {student_row}")

        if student_correct:
            selected_reasoning, selected_strategy = _normalize_boxed_answer(selected_reasoning, answer)
        else:
            selected_strategy = student_strategy or "wrong answer"

        merged_row = dict(student_row)
        merged_row["answer"] = answer
        merged_row["boxed_answer"] = answer
        merged_row["selected_reasoning"] = selected_reasoning
        merged_row["reasoning"] = selected_reasoning
        merged_row["actual_answer"] = selected_reasoning
        merged_row["source_reasoning"] = source
        merged_row["student_correct"] = student_correct
        merged_row["student_reasoning"] = _get_first(student_row, REASONING_KEYS)
        merged_row["teacher_reasoning"] = _get_first(teacher_row, REASONING_KEYS)
        merged_row["selected_strategy"] = selected_strategy
        merged_row["messages"] = _build_messages(prompt, selected_reasoning)
        merged_rows.append(merged_row)

    _write_jsonl(output_path, merged_rows)
    total = len(merged_rows)
    print(f"Wrote {output_path}")
    print(f"Student correct: {student_correct_count}/{total} ({student_correct_count / total:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())