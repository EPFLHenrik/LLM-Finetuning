"""Remove gold_fallback examples from *_distilled.jsonl files.

For each *_distilled.jsonl:
  - keeps examples where distill.mode != "gold_fallback"
  - overwrites the original file with the cleaned version

All removed examples are written to a single output file.

Usage:

    # Apply
    python multilingual/data/distilled/remove_gold_fallback.py \
        --distilled_dir multilingual/data/distilled/v2_fallback/
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=Path, required=True)
    parser.add_argument("--removed_output", type=Path, default=None,
                        help="Path for removed examples. Default: <distilled_dir>/all_removed_gold_fallback.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    removed_output = args.removed_output or (
        args.distilled_dir / "all_removed_gold_fallback.jsonl"
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

                if ex.get("distill", {}).get("mode") == "gold_fallback":
                    ex["_source_file"] = str(path)
                    removed.append(ex)
                else:
                    kept.append(ex)

        print(f"  {path.name}: kept={len(kept)}, removed={len(removed)}")
        total_kept    += len(kept)
        total_removed += len(removed)
        all_removed.extend(removed)

        if dry_run := args.dry_run:
            continue
        
        # rewrite distilled files
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