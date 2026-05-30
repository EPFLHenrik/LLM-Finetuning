"""Remove gold_fallback examples from *_distilled.jsonl files.

Removes only gold_fallback examples where:
  - any Llama independent attempt was NOT_SURE, AND
  - Apertus got a wrong answer (not NOT_SURE, just wrong)

This targets cases where the teacher was genuinely uncertain before
resorting to gold fallback, producing unreliable rationales.

Usage:
    # Dry run
    python multilingual/data/remove_gold_fallback.py \
        --distilled_dir multilingual/data/distilled/v2_fallback/sft_1 \
        --dry_run

    # Apply
    python multilingual/data/remove_gold_fallback.py \
        --distilled_dir multilingual/data/distilled/v2_fallback/sft_1
"""

import argparse
import json
from pathlib import Path

TEACHER_PRIMARY  = "meta-llama/Llama-3.3-70B-Instruct"
TEACHER_FALLBACK = "swiss-ai/Apertus-70B-Instruct-2509"


def _should_remove(ex: dict) -> bool:
    distill = ex.get("distill", {})
    if distill.get("mode") != "gold_fallback":
        return False

    attempts = distill.get("attempts", [])
    independent = [a for a in attempts if a.get("stage") == "independent"]

    # Any Llama independent attempt was NOT_SURE
    any_llama_not_sure = any(
        a.get("model") == TEACHER_PRIMARY and
        a.get("invalid_reason") == "not_sure"
        for a in independent
    )
    if not any_llama_not_sure:
        return False

    # Apertus got a wrong answer (invalid but not NOT_SURE)
    last_apertus = next(
        (a for a in reversed(independent) if a.get("model") == TEACHER_FALLBACK),
        None,
    )
    apertus_wrong = (
        last_apertus is not None and
        last_apertus.get("invalid_reason") != "not_sure" and
        last_apertus.get("invalid_reason") is not None
    )
    return apertus_wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=Path, required=True)
    parser.add_argument("--removed_output", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    removed_output = args.removed_output or (
        args.distilled_dir / "all_removed_gold_fallback_v3.jsonl"
    )

    files = sorted(args.distilled_dir.rglob("*_distilled.jsonl"))
    print(f"Found {len(files)} *_distilled.jsonl files")
    if args.dry_run:
        print("[DRY RUN] No files will be modified.\n")

    all_removed = []
    total_kept = total_removed = 0

    for path in files:
        kept, removed = [], []

        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue

                if _should_remove(ex):
                    ex["_source_file"] = str(path)
                    removed.append(ex)
                else:
                    kept.append(ex)

        print(f"  {path.name}: kept={len(kept)}, removed={len(removed)}")
        total_kept    += len(kept)
        total_removed += len(removed)
        all_removed.extend(removed)

        if args.dry_run:
            continue

        with path.open("w", encoding="utf-8") as f:
            for ex in kept:
                if isinstance(ex, str):
                    f.write(ex if ex.endswith("\n") else ex + "\n")
                else:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTotal kept:    {total_kept}")
    print(f"Total removed: {total_removed}")

    if args.dry_run:
        return

    removed_output.parent.mkdir(parents=True, exist_ok=True)
    with removed_output.open("a", encoding="utf-8") as f:
        for ex in all_removed:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Removed examples saved to: {removed_output}")


if __name__ == "__main__":
    main()