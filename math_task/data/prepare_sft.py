"""Prepare SFT data from `open-r1/OpenR1-Math-220k`.

Produces a JSONL of chat-format examples ready for TRL SFTTrainer:
    {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"<think>...</think>\\n\\n...\\boxed{42}"}], "source": "..."}

Filtering pipeline (mirrors LIMO/LIMR/Open-R1's recommended practices):
  1. Use only the `default` split (94K) — the README says it outperforms `extended` after SFT.
  2. Drop rows where no R1 generation was verified correct (correctness_math_verify == 0).
  3. Pick the first correct generation per row as the assistant content.
  4. Require the generation contains `<think>...</think>` (Qwen3 thinking format).
  5. Require the generation ends with an extractable `\\boxed{...}` answer.
  6. Length filter: 1500 <= Qwen3 tokens <= 8000. Lower bound drops trivial traces;
     upper bound keeps everything well under the 8192 SFT training context.
  7. Stratified sampling across the `source` field so we don't end up 90% one subset.

Usage:
    python -m math_task.data.prepare_sft \\
        --n-total 5000 \\
        --output /scratch/<user>/sft_data/openr1_math_5k.jsonl

Inspect stats after:
    python -m math_task.data.prepare_sft --stats /scratch/<user>/sft_data/openr1_math_5k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# We intentionally lazy-import datasets / transformers so this module is
# importable without those heavy deps installed (e.g., for unit tests).


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

DEFAULT_SOURCE = "open-r1/OpenR1-Math-220k"
DEFAULT_SPLIT = "default"        # the 94K curated subset
DEFAULT_N_TOTAL = 5000
DEFAULT_MIN_TOKENS = 1500
DEFAULT_MAX_TOKENS = 8000


def _extract_boxed(text: str) -> Optional[str]:
    """Return the inner content of the LAST \\boxed{...} in `text`, or None."""
    # Import the project's grader so we use the SAME extraction logic as eval.
    from evaluate.extract_answer import extract_boxed_answer
    return extract_boxed_answer(text, strip_double_curly_brace=True)


def _has_think_block(text: str) -> bool:
    """Quick structural check: needs both an opening and closing think tag."""
    return THINK_OPEN in text and THINK_CLOSE in text


def _pick_correct_generation(row: dict) -> Optional[str]:
    """Return the first generation in `row` whose final answer was verified
    correct by Math-Verify. Returns None if no correct generation exists.

    The OpenR1-Math-220k schema has:
      - row["generations"]: list[str]  -- the R1 reasoning traces
      - row["correctness_math_verify"]: list[bool]  -- per-generation verdicts
    """
    gens = row.get("generations") or []
    flags = row.get("correctness_math_verify") or []
    if not gens or not flags or len(gens) != len(flags):
        return None
    for gen, ok in zip(gens, flags):
        if ok:
            return gen
    return None


def _qwen3_tokenizer():
    """Cached Qwen3 tokenizer for length-filtering."""
    from transformers import AutoTokenizer
    if not hasattr(_qwen3_tokenizer, "_tok"):
        _qwen3_tokenizer._tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    return _qwen3_tokenizer._tok


def _token_len(text: str) -> int:
    tok = _qwen3_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def prepare(
    n_total: int = DEFAULT_N_TOTAL,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = 42,
    output_path: Optional[Path] = None,
    hf_dataset: str = DEFAULT_SOURCE,
    hf_split: str = DEFAULT_SPLIT,
) -> List[Dict]:
    """Build the SFT JSONL and return the list of examples.

    If `output_path` is given, also writes JSONL to disk.
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    print(f"[load] {hf_dataset} split={hf_split}")
    ds = load_dataset(hf_dataset, hf_split, split="train")
    print(f"[load] {len(ds)} rows")

    # Bucket by source so we can stratify-sample at the end.
    by_source: Dict[str, List[Dict]] = {}
    dropped = Counter()

    for i, row in enumerate(ds):
        if i and i % 10000 == 0:
            print(f"[filter] processed {i}/{len(ds)} rows  kept={sum(len(v) for v in by_source.values())}")

        gen = _pick_correct_generation(row)
        if gen is None:
            dropped["no_correct_gen"] += 1
            continue
        if not _has_think_block(gen):
            dropped["no_think_block"] += 1
            continue
        if _extract_boxed(gen) is None:
            dropped["no_boxed_answer"] += 1
            continue

        n_tok = _token_len(gen)
        if n_tok < min_tokens:
            dropped["too_short"] += 1
            continue
        if n_tok > max_tokens:
            dropped["too_long"] += 1
            continue

        source = row.get("source") or "unknown"
        problem = row.get("problem") or row.get("question") or ""
        if not problem.strip():
            dropped["empty_problem"] += 1
            continue

        example = {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": gen},
            ],
            "source": source,
            "n_tokens_assistant": n_tok,
        }
        by_source.setdefault(source, []).append(example)

    total_kept = sum(len(v) for v in by_source.values())
    print(f"\n[filter] {total_kept} rows passed all filters")
    print(f"[filter] dropped counts: {dict(dropped)}")
    print(f"[filter] surviving sources: {[(s, len(v)) for s, v in by_source.items()]}")

    # Stratified sample: take roughly n_total / num_sources per source,
    # capped at availability. Smaller-source remainder reallocated to larger.
    sources = sorted(by_source.keys())
    target_per_source = n_total // max(len(sources), 1)
    sampled: List[Dict] = []
    leftover_quota = 0

    for s in sources:
        pool = by_source[s]
        rng.shuffle(pool)
        take = min(target_per_source, len(pool))
        sampled.extend(pool[:take])
        leftover_quota += (target_per_source - take)

    # Re-distribute unmet quota across sources that still have capacity.
    while leftover_quota > 0:
        progress = False
        for s in sources:
            if leftover_quota <= 0:
                break
            pool = by_source[s]
            already = sum(1 for ex in sampled if ex["source"] == s)
            if already < len(pool):
                sampled.append(pool[already])
                leftover_quota -= 1
                progress = True
        if not progress:
            break  # all sources exhausted

    rng.shuffle(sampled)
    sampled = sampled[:n_total]

    print(f"\n[sample] selected {len(sampled)} / {n_total} examples")
    print(f"[sample] per-source counts: {Counter(ex['source'] for ex in sampled).most_common()}")
    print(f"[sample] token length p50/p90/p99: "
          f"{_percentile([ex['n_tokens_assistant'] for ex in sampled], 50)}/"
          f"{_percentile([ex['n_tokens_assistant'] for ex in sampled], 90)}/"
          f"{_percentile([ex['n_tokens_assistant'] for ex in sampled], 99)}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in sampled:
                # Drop the n_tokens_assistant field from the written record —
                # SFTTrainer only needs `messages` (and optionally `source`).
                rec = {k: v for k, v in ex.items() if k != "n_tokens_assistant"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[save] {output_path}")

    return sampled


def _percentile(xs: List[int], p: int) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, (len(s) * p) // 100))
    return s[idx]


def stats(jsonl_path: Path) -> None:
    """Quick stats printout for an existing prepared JSONL."""
    examples = []
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

    # Recompute token counts using the tokenizer.
    lens = [_token_len(ex["messages"][1]["content"]) for ex in examples[:500]]
    print(f"[stats] assistant tokens p50/p90/p99 (first 500): "
          f"{_percentile(lens, 50)}/{_percentile(lens, 90)}/{_percentile(lens, 99)}")

    # Sanity check the format of a random example.
    sample = examples[0]
    print("\n[stats] sample[0]:")
    print(f"  user: {sample['messages'][0]['content'][:120]}...")
    asst = sample['messages'][1]['content']
    print(f"  assistant (first 200): {asst[:200]}...")
    print(f"  assistant (last 200):  ...{asst[-200:]}")
    print(f"  ends with boxed? {_extract_boxed(asst) is not None}")
    print(f"  has <think>? {THINK_OPEN in asst and THINK_CLOSE in asst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=None,
                    help="If given, print stats for an already-prepared JSONL and exit.")
    ap.add_argument("--n-total", type=int, default=DEFAULT_N_TOTAL,
                    help=f"Target number of examples (default {DEFAULT_N_TOTAL}).")
    ap.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
                    help=f"Minimum assistant trace length in Qwen3 tokens (default {DEFAULT_MIN_TOKENS}).")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"Maximum assistant trace length in Qwen3 tokens (default {DEFAULT_MAX_TOKENS}).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=False,
                    help="Output JSONL path. Defaults to /scratch/<USER>/sft_data/openr1_math_<N>k.jsonl")
    args = ap.parse_args()

    if args.stats is not None:
        stats(args.stats)
        return

    output = args.output
    if output is None:
        import os
        user = os.environ.get("USER", "user")
        output = Path(f"/scratch/{user}/sft_data/openr1_math_{args.n_total // 1000}k.jsonl")

    prepare(
        n_total=args.n_total,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        output_path=output,
    )


if __name__ == "__main__":
    main()
