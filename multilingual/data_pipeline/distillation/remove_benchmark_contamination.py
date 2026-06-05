"""multilingual/data/distilled/remove_benchmark_contamination.py

Removes benchmark examples from all *_distilled.jsonl files in sft_1/
by matching source_id against the benchmark set.

Usage:
    python multilingual/data/distilled/remove_benchmark_contamination.py \
        --benchmark multilingual/data/benchmarks/benchmark_all.jsonl \
        --sft1_dir multilingual/data/distilled/v2_fallback/sft_1 \
        --dry_run
"""

import argparse
import json
from pathlib import Path


def load_benchmark_ids(path: Path) -> set:
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ids.add(r.get("source_id", ""))
    return ids


def process_file(path: Path, benchmark_ids: set, dry_run: bool) -> dict:
    records = []
    removed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("source_id", "") in benchmark_ids:
                    removed += 1
                else:
                    records.append(r)

    if not dry_run and removed > 0:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"kept": len(records), "removed": removed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path,
                        default=Path("multilingual/data/benchmarks/benchmark_all.jsonl"))
    parser.add_argument("--sft1_dir", type=Path,
                        default=Path("multilingual/data/distilled/v2_fallback/sft_1"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    benchmark_ids = load_benchmark_ids(args.benchmark)
    print(f"Loaded {len(benchmark_ids)} benchmark IDs")

    files = sorted(args.sft1_dir.glob("*_distilled.jsonl"))
    print(f"Found {len(files)} distilled files in {args.sft1_dir}")
    if args.dry_run:
        print("[DRY RUN] No files will be modified\n")

    total_removed = total_kept = 0
    for f in files:
        stats = process_file(f, benchmark_ids, args.dry_run)
        if stats["removed"] > 0:
            print(f"  {f.name}: removed={stats['removed']}, kept={stats['kept']}")
        total_removed += stats["removed"]
        total_kept += stats["kept"]

    print(f"\nTotal removed: {total_removed} | Total kept: {total_kept}")
    if args.dry_run:
        print("[DRY RUN] No files written.")


if __name__ == "__main__":
    main()