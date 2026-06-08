#!/usr/bin/env python3
"""Filter generate_answers.py outputs to the partial-success zone for GRPO.

Given N runs of generate_answers.py on the same dataset, keep only the prompts
that were answered correctly 1 or 2 times (i.e. min_correct <= correct_count
<= max_correct). These are the prompts the model can sometimes get right —
the optimal training signal for GRPO.

Usage:
    python scripts/filter_partial_success.py run1.jsonl run2.jsonl run3.jsonl \
        --output data/preprocessing/sft_partial_success.jsonl

    # Or pass a folder containing the run files:
    python scripts/filter_partial_success.py runs/ \
        --output data/preprocessing/sft_partial_success.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _is_correct(row: dict) -> bool:
    val = row.get("correct")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return False


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: {path}:{lineno}: skipping invalid JSON: {e}", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract partial-success prompts for GRPO training.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="JSONL files from generate_answers.py, or a single folder containing them.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file (prompt + answer rows).")
    parser.add_argument("--min-correct", type=int, default=1, help="Min correct runs to keep a prompt (default: 1).")
    parser.add_argument("--max-correct", type=int, default=2, help="Max correct runs to keep a prompt (default: 2).")
    args = parser.parse_args(argv)

    # Resolve input files
    input_files: list[Path] = []
    for inp in args.inputs:
        if inp.is_dir():
            input_files.extend(sorted(inp.glob("*.jsonl")))
        elif inp.is_file():
            input_files.append(inp)
        else:
            print(f"Error: not a file or directory: {inp}", file=sys.stderr)
            return 1

    if not input_files:
        print("Error: no input files found.", file=sys.stderr)
        return 1

    n_runs = len(input_files)
    print(f"Loading {n_runs} run(s): {[f.name for f in input_files]}")

    if not (args.min_correct <= args.max_correct <= n_runs):
        print(f"Error: need min_correct ({args.min_correct}) <= max_correct ({args.max_correct}) <= n_runs ({n_runs})", file=sys.stderr)
        return 1

    # Group by prompt: track correct count and canonical answer
    # Key: prompt text. Value: {correct_count, answer, seen_in_runs}
    stats: dict[str, dict] = defaultdict(lambda: {"correct_count": 0, "answer": None, "run_count": 0})

    for run_idx, path in enumerate(input_files):
        rows = _load_jsonl(path)
        print(f"  run {run_idx + 1}: {len(rows)} rows from {path.name}")
        for row in rows:
            prompt = row.get("prompt", "")
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt)
            entry = stats[prompt]
            entry["run_count"] += 1
            if entry["answer"] is None:
                entry["answer"] = row.get("answer", "")
            if _is_correct(row):
                entry["correct_count"] += 1

    # Filter to partial-success zone
    kept = [
        {"prompt": prompt, "answer": entry["answer"]}
        for prompt, entry in stats.items()
        if args.min_correct <= entry["correct_count"] <= args.max_correct
        and entry["run_count"] == n_runs  # only keep prompts present in all runs
    ]

    # Report
    total = len(stats)
    always_correct = sum(1 for e in stats.values() if e["correct_count"] == n_runs)
    always_wrong   = sum(1 for e in stats.values() if e["correct_count"] == 0)
    partial        = len(kept)
    print(f"\nPrompt breakdown across {n_runs} runs:")
    print(f"  Always correct ({n_runs}/{n_runs}): {always_correct:>6}  ({always_correct/total:.1%}) — skip, already learned")
    print(f"  Always wrong   (0/{n_runs}):  {always_wrong:>6}  ({always_wrong/total:.1%}) — skip, too hard for RL")
    print(f"  Partial success ({args.min_correct}-{args.max_correct}/{n_runs}): {partial:>6}  ({partial/total:.1%}) — kept for GRPO")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {partial} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())