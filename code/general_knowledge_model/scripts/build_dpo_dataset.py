#!/usr/bin/env python3
"""Build DPO training pairs from student run files and teacher distillation data.

For each prompt where the student answered incorrectly, we pair:
  chosen  — the teacher's correct completion (from merged_distillation.jsonl)
  rejected — the student's wrong completion (from generate_answers.py run files)

Usage:
    python scripts/build_dpo_dataset.py \
        --runs data/preprocessing/grpo_runs/run1.jsonl \
               data/preprocessing/grpo_runs/run2.jsonl \
               data/preprocessing/grpo_runs/run3.jsonl \
        --distillation data/preprocessing/merged_distillation.jsonl \
        --output data/preprocessing/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Warning: {path}:{lineno}: skipping invalid JSON: {exc}", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DPO pairs from student runs + teacher distillation.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="generate_answers.py output files (must include completion and correct fields)")
    parser.add_argument("--distillation", type=Path,
                        default=Path("data/general_knowledge/processed/train_datasets/merged_distillations.jsonl"),
                        help="merged_distillations.jsonl (must include messages with assistant CoT turn)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file with (prompt, chosen, rejected, answer) rows")
    args = parser.parse_args(argv)

    # Load teacher distillation: keyed by prompt text
    print(f"Loading distillation from {args.distillation} ...")
    dist_rows = _load_jsonl(args.distillation)
    teacher_by_prompt: dict[str, str] = {}
    for r in dist_rows:
        prompt = r.get("prompt", "")
        if not prompt:
            continue
        # Prefer messages field assistant turn; fall back to completion field
        msgs = r.get("messages", [])
        asst_content = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
        if asst_content is None:
            asst_content = r.get("completion", "")
        if asst_content:
            teacher_by_prompt[prompt] = asst_content
    print(f"  {len(teacher_by_prompt)} teacher completions indexed")

    # Load run files, collect one wrong completion per prompt (first wrong seen)
    wrong_by_prompt: dict[str, dict] = {}
    for run_path in args.runs:
        print(f"Loading run {run_path} ...")
        run_rows = _load_jsonl(run_path)
        for r in run_rows:
            prompt = r.get("prompt", "")
            if not prompt or prompt in wrong_by_prompt:
                continue
            correct = r.get("correct")
            if isinstance(correct, str):
                correct = correct.strip().lower() == "true"
            if not correct:
                completion = r.get("completion", "")
                if completion:
                    wrong_by_prompt[prompt] = {
                        "completion": completion,
                        "answer": r.get("answer", ""),
                    }
    print(f"  {len(wrong_by_prompt)} unique wrong-answer prompts collected")

    # Build pairs
    pairs = []
    missing_teacher = 0
    for prompt, wrong in wrong_by_prompt.items():
        chosen = teacher_by_prompt.get(prompt)
        if chosen is None:
            missing_teacher += 1
            continue
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": wrong["completion"],
            "answer": wrong["answer"],
        })

    print(f"\nPair breakdown:")
    print(f"  Wrong completions found:          {len(wrong_by_prompt)}")
    print(f"  Missing teacher completion:       {missing_teacher}")
    print(f"  Final DPO pairs:                  {len(pairs)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(pairs)} pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
