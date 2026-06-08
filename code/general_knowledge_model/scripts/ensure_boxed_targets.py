#!/usr/bin/env python3
"""Ensure assistant messages contain boxed reference answers.

Reads a JSONL/JSON file of rows (conversational or flat) and writes a
conversational JSONL where each row has a `messages` array and the final
assistant message content is the boxed reference (e.g. "\\boxed{A}").

This is intended to be used before running `training/lora_post_train.py` so
that supervised LoRA training learns to produce boxed answers by imitation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROMPT_KEYS = ("prompt", "question", "input", "text")
ANSWER_KEYS = ("answer", "reference", "label", "target", "answerKey")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise SystemExit(f"{path}: JSON must contain a list of objects")
    else:
        raise SystemExit("Unsupported extension, use .jsonl or .json")


def _get_first(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v:
                return v
    return None


def ensure_boxed_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        # try to find conversational `messages` already
        messages = r.get("messages")
        if not isinstance(messages, list) or not messages:
            # build a messages array from available fields
            system = r.get("system")
            prompt = _get_first(r, PROMPT_KEYS)
            messages = []
            if system:
                messages.append({"role": "system", "content": str(system)})
            if prompt:
                messages.append({"role": "user", "content": str(prompt)})
        else:
            # copy shallowly to avoid mutating input
            messages = [dict(m) for m in messages]

        reference = _get_first(r, ANSWER_KEYS)
        if reference is None:
            # if no reference, preserve original assistant if exists, else skip
            has_assistant = any(m.get("role") == "assistant" for m in messages)
            if not has_assistant:
                # append empty assistant placeholder
                messages.append({"role": "assistant", "content": ""})
        else:
            boxed = f"\\boxed{{{reference}}}"
            # find last assistant message index
            last_assistant_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break
            if last_assistant_idx is not None:
                messages[last_assistant_idx]["content"] = boxed
            else:
                messages.append({"role": "assistant", "content": boxed})

        new_row = dict(r)
        new_row["messages"] = messages
        # remove top-level prompt/answer to avoid ambiguity downstream
        for k in list(new_row.keys()):
            if k in PROMPT_KEYS + ANSWER_KEYS:
                new_row.pop(k, None)
        out.append(new_row)
    return out


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="Input .jsonl or .json file")
    p.add_argument("--output", required=True, type=Path, help="Output .jsonl file (conversational)")
    p.add_argument("--dry-run", action="store_true", help="Print a sample and don't write output")
    args = p.parse_args()

    rows = _load_rows(args.input)
    new_rows = ensure_boxed_messages(rows)

    if args.dry_run:
        print("Sample converted example:")
        import pprint

        pprint.pprint(new_rows[0])
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for r in new_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(new_rows)} rows to {args.output}")


if __name__ == "__main__":
    cli()
