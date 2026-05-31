"""multilingual/data/distilled/restore_gold_fallback.py

Restores valid gold fallback examples that were incorrectly removed from
_distilled.jsonl files. Reads _all.jsonl to find valid examples missing
from _distilled.jsonl, checks they don't match the NOT_SURE removal
condition, and writes the merged result to sft_2/.

NOT_SURE removal condition (from remove_gold_fallback_v3.py):
  Skip gold fallback if:
    - Apertus was NOT_SURE, OR
    - Any Llama attempt was NOT_SURE AND Apertus got wrong answer

Usage:
    python multilingual/data/distilled/restore_gold_fallback.py \
        --distill_dir multilingual/data/distilled/v2_fallback \
        --output_dir multilingual/data/distilled/v2_fallback/sft_2 \
        --dry_run
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_source_ids(path: Path) -> Set[str]:
    ids = set()
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    ids.add(r.get("source_id", ""))
                except json.JSONDecodeError:
                    pass
    return ids


def _should_have_been_removed(r: Dict) -> bool:
    """Return True if this gold fallback example matches the NOT_SURE removal condition.

    Remove if:
      - Apertus was NOT_SURE, OR
      - Any Llama independent attempt was NOT_SURE (regardless of Apertus)
    """
    distill = r.get("distill", {})
    if distill.get("mode") != "gold_fallback":
        return False

    attempts = distill.get("attempts", [])

    # Apertus said NOT_SURE
    if any(
        a.get("model", "").startswith("swiss-ai") and
        a.get("invalid_reason") == "not_sure"
        for a in attempts
    ):
        return True

    # Any Llama independent attempt was NOT_SURE
    if any(
        a.get("model", "").startswith("meta-llama") and
        a.get("stage") == "independent" and
        a.get("invalid_reason") == "not_sure"
        for a in attempts
    ):
        return True

    return False


def process_file(
    all_path: Path,
    distilled_path: Path,
    output_path: Path,
    dry_run: bool,
    global_seen: set,
) -> Dict:
    if not all_path.exists():
        return {"skipped": True, "reason": "no _all.jsonl"}

    all_records = _load_jsonl(all_path)
    existing_ids = _load_source_ids(distilled_path)

    # Find valid examples missing from distilled that should not have been removed
    to_restore = []
    for r in all_records:
        distill = r.get("distill", {})
        if not distill.get("valid", False):
            continue
        sid = r.get("source_id", "")
        if sid in existing_ids:
            continue
        if _should_have_been_removed(r):
            continue
        to_restore.append(r)

    # Load existing distilled records
    existing_records = _load_jsonl(distilled_path) if distilled_path.exists() else []

    # Deduplicate by source_id globally across all files
    merged = []
    for r in existing_records + to_restore:
        sid = r.get("source_id", "")
        if sid not in global_seen:
            global_seen.add(sid)
            merged.append(r)

    stats = {
        "existing": len(existing_records),
        "restored": len(to_restore),
        "total": len(merged),
        "all_valid": sum(1 for r in all_records if r.get("distill", {}).get("valid")),
    }

    if not dry_run and merged:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill_dir", type=Path,
                        default=Path("multilingual/data/distilled/v2_fallback"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("multilingual/data/distilled/v2_fallback/sft_2"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Find all _all.jsonl files
    all_files = sorted(args.distill_dir.glob("*_all.jsonl"))
    # Exclude shuffle meta and non-distillation files
    all_files = [f for f in all_files if "shuffle_meta" not in f.name]

    print(f"Found {len(all_files)} _all.jsonl files")
    print(f"Output dir: {args.output_dir}")
    if args.dry_run:
        print("[DRY RUN] No files will be written\n")

    total_existing = total_restored = total_out = 0

    global_seen: Set[str] = set()
    for all_path in all_files:
        base = all_path.name.replace("_all.jsonl", "")
        distilled_path = args.distill_dir / f"{base}_distilled.jsonl"
        output_path = args.output_dir / f"{base}_distilled.jsonl"

        stats = process_file(all_path, distilled_path, output_path, args.dry_run, global_seen)

        if stats.get("skipped"):
            continue

        restored = stats["restored"]
        existing = stats["existing"]
        total = stats["total"]

        if restored > 0 or existing > 0:
            print(f"  {base}: existing={existing}, restored={restored} → total={total}")

        total_existing += existing
        total_restored += restored
        total_out += total

    print(f"\nTotal: existing={total_existing}, restored={total_restored} → total={total_out}")
    if args.dry_run:
        print("[DRY RUN] No files written.")


if __name__ == "__main__":
    main()