#!/usr/bin/env python3
"""Merge corrected student/teacher files into chat-style JSONL.

The script reads two JSONL files with matching prompt/answer pairs and writes
rows containing only:
prompt, answer, messages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT_KEYS = ("prompt", "question", "input", "text")
ANSWER_KEYS = ("gold_answer", "answer", "reference", "label", "target", "answerKey", "boxed_answer")


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
            print(f"{path}:{index}: duplicate row key for prompt/answer pair")
            pass
        indexed[key] = row
    return indexed


def _pick_field(row: dict[str, Any], primary: str, fallback: str | None = None) -> str:
    value = row.get(primary)
    if value is None and fallback is not None:
        value = row.get(fallback)
    if value is None:
        return ""
    return str(value)


def _pick_bool_field(row: dict[str, Any], primary: str, fallback: str | None = None) -> bool | None:
    value = row.get(primary)
    if value is None and fallback is not None:
        value = row.get(fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "0", "no", "n"}:
            return False
    return None


def _build_messages(prompt: str, system_message: str, completion: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge corrected student/teacher JSONL files.")
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
    wrong_rows: list[dict[str, Any]] = []
    dropped_both_wrong = 0

    for key, student_row in student_rows.items():
        teacher_row = teacher_rows[key]
        prompt = _get_first(student_row, PROMPT_KEYS)
        assert prompt is not None
        answer = _get_first(student_row, ANSWER_KEYS) or _get_first(teacher_row, ANSWER_KEYS)
        if answer is None:
            raise SystemExit(f"Missing answer field for row: {student_row}")

        student_correct = _pick_bool_field(student_row, "correct")
        trainer_correct = _pick_bool_field(teacher_row, "correct", fallback="teacher_correct")

        # Drop rows where both sides are explicitly incorrect.
        if student_correct is False and trainer_correct is False:
            dropped_both_wrong += 1
            wrong_rows.append({"prompt": prompt, "answer": answer or ""})
            continue

        student_completion = _pick_field(student_row, "completion")
        trainer_completion = _pick_field(teacher_row, "completion", fallback="teacher_completion")
        system_message = _pick_field(student_row, "system_message")
        if not system_message:
            system_message = _pick_field(teacher_row, "system_message", fallback="teacher_system_message")

        chosen_completion = student_completion if student_correct is True else trainer_completion
        merged_row = {
            "prompt": prompt,
            "answer": answer,
            "messages": _build_messages(prompt, system_message, chosen_completion),
        }
        merged_rows.append(merged_row)

    _write_jsonl(output_path, merged_rows)
    print(f"Wrote {output_path}")
    print(f"Rows: {len(merged_rows)}")
    print(f"Dropped both-wrong rows: {dropped_both_wrong}")

    if wrong_rows:
        wrong_path = Path(__file__).resolve().parents[1] / "data" / "preprocessing" / "wrong_distillations.jsonl"
        existing: list[dict[str, Any]] = []
        if wrong_path.exists() and wrong_path.stat().st_size > 0:
            try:
                existing = _read_jsonl(wrong_path)
            except SystemExit:
                existing = []
        seen = {(r["prompt"], r["answer"]) for r in existing}
        new_wrong = [r for r in wrong_rows if (r["prompt"], r["answer"]) not in seen]
        _write_jsonl(wrong_path, existing + new_wrong)
        print(f"Appended {len(new_wrong)} new both-wrong rows to {wrong_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())