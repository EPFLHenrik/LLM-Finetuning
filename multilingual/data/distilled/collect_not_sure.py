"""Collect all NOT_SURE examples from *_distilled.jsonl files into one file.
Counts unique examples by source_id. One example = one JSONL line.

Usage:
    python multilingual/data/distilled/collect_not_sure.py \
        --distilled_dir multilingual/data/distilled/v2_fallback \
        --output multilingual/data/distilled/v2_fallback/not_sure_all.jsonl \
        --dry_run
"""

import argparse
import json
from pathlib import Path


def _is_not_sure(ex: dict) -> bool:
    distill = ex.get("distill", {})
    if distill.get("invalid_reason") == "not_sure":
        return True
    if distill.get("mode") == "gold_fallback":
        return True
    for attempt in distill.get("attempts", []):
        if attempt.get("invalid_reason") == "not_sure":
            return True
        raw   = str(attempt.get("raw",   "") or "")
        clean = str(attempt.get("clean", "") or "")
        if "NOT_SURE" in raw.upper() or "NOT_SURE" in clean.upper():
            return True
    return False


def _is_gold_fallback(ex: dict) -> bool:
    return ex.get("distill", {}).get("mode") == "gold_fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("not_sure_all.jsonl"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    files = sorted(args.distilled_dir.rglob("*_distilled.jsonl"))
    print(f"Found {len(files)} *_distilled.jsonl files\n")

    collected = []
    seen_ids  = set()
    gold_fallback_count = 0
    not_sure_only_count = 0

    for path in files:
        file_count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not _is_not_sure(ex):
                    continue

                sid = str(ex.get("source_id") or ex.get("id") or "")
                if sid and sid in seen_ids:
                    continue  # skip duplicate source_ids
                if sid:
                    seen_ids.add(sid)

                ex["_source_file"] = str(path)
                collected.append(ex)
                file_count += 1

                if _is_gold_fallback(ex):
                    gold_fallback_count += 1
                else:
                    not_sure_only_count += 1

        if file_count:
            print(f"  {path.name}: {file_count} unique NOT_SURE examples")

    print(f"\nTotal unique NOT_SURE examples: {len(collected)}")
    print(f"  gold_fallback:                {gold_fallback_count}")
    print(f"  not_sure in early attempt:    {not_sure_only_count}")

    if args.dry_run:
        print("\n[DRY RUN] Not writing output.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in collected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()