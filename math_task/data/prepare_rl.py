"""Prepare RL data from `agentica-org/DeepScaleR-Preview-Dataset`.

Produces a JSONL of {prompt, answer} pairs ready for ms-swift GRPO/CISPO:
    {"messages": [{"role": "user", "content": "<problem>"}],
     "solution": "<gold answer>",
     "source": "..."}

The dataset already deduplicates AIME 1984-2023 + AMC pre-2023 + Omni-MATH + Still
against AIME24/25 and MATH500, and was curated against a 1.5B model's difficulty
band -- almost exactly the right band for our 1.7B base. No additional filtering
is done here; difficulty filtering happens downstream in `filter_learnability.py`.

Usage:
    python -m math_task.data.prepare_rl \\
        --output /scratch/<user>/rl_data/deepscaler_full.jsonl

Inspect after:
    python -m math_task.data.prepare_rl --stats /scratch/<user>/rl_data/deepscaler_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_SOURCE = "agentica-org/DeepScaleR-Preview-Dataset"
DEFAULT_SPLIT = "train"


def _normalize(row: dict, source_tag: str) -> Optional[Dict]:
    """Map a DeepScaleR row to our RL training schema.

    DeepScaleR schema: {"problem": str, "answer": str, "solution": str (optional)}
    Output: ms-swift GRPO-compatible record with `solution` field used by the
    `accuracy` reward function.
    """
    problem = (row.get("problem") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not problem or not answer:
        return None
    return {
        "messages": [{"role": "user", "content": problem}],
        "solution": answer,
        "source": source_tag,
    }


def prepare(
    output_path: Path,
    hf_dataset: str = DEFAULT_SOURCE,
    hf_split: str = DEFAULT_SPLIT,
    limit: Optional[int] = None,
) -> List[Dict]:
    from datasets import load_dataset

    print(f"[load] {hf_dataset} split={hf_split}")
    ds = load_dataset(hf_dataset, split=hf_split)
    print(f"[load] {len(ds)} rows")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"[load] truncated to {len(ds)}")

    out: List[Dict] = []
    dropped = Counter()
    for i, row in enumerate(ds):
        if i and i % 10000 == 0:
            print(f"[normalize] {i}/{len(ds)}  kept={len(out)}")
        rec = _normalize(row, source_tag=hf_dataset)
        if rec is None:
            dropped["empty_problem_or_answer"] += 1
            continue
        out.append(rec)

    print(f"\n[normalize] {len(out)} records kept, dropped={dict(dropped)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] {output_path}")
    return out


def stats(jsonl_path: Path) -> None:
    examples: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if not examples:
        print(f"(empty file: {jsonl_path})")
        return
    print(f"[stats] {len(examples)} examples")
    print(f"[stats] per-source: {Counter(ex['source'] for ex in examples).most_common()}")
    print("\n[stats] sample[0]:")
    print(f"  prompt:   {examples[0]['messages'][0]['content'][:160]}...")
    print(f"  solution: {examples[0]['solution']}")
    plens = [len(ex["messages"][0]["content"]) for ex in examples]
    print(f"[stats] prompt char-len p50/p90/p99: "
          f"{sorted(plens)[len(plens)//2]}/"
          f"{sorted(plens)[(len(plens)*9)//10]}/"
          f"{sorted(plens)[(len(plens)*99)//100]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=None,
                    help="If given, print stats for an already-prepared JSONL and exit.")
    ap.add_argument("--output", type=Path, required=False,
                    help="Output JSONL path. Defaults to /scratch/<USER>/rl_data/deepscaler_full.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: cap at N rows.")
    args = ap.parse_args()

    if args.stats is not None:
        stats(args.stats)
        return

    output = args.output
    if output is None:
        user = os.environ.get("USER", "user")
        output = Path(f"/scratch/{user}/rl_data/deepscaler_full.jsonl")

    prepare(output_path=output, limit=args.limit)


if __name__ == "__main__":
    main()
