"""Split removed_distilled_benchmark_examples.jsonl into per-language files.

Usage:
    python multilingual/eval/split_benchmark_by_language.py \\
        --input multilingual/data/distilled/v2_fallback/removed_distilled_benchmark_examples.jsonl \\
        --output_dir multilingual/data/benchmarks
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("multilingual/data/benchmarks"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["split"] = "test"
                records.append(r)

    # Combined file
    combined = args.output_dir / "benchmark_all.jsonl"
    with open(combined, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} records to {combined}")

    # Per-language files
    by_lang = defaultdict(list)
    for r in records:
        by_lang[r.get("language", "unknown")].append(r)

    for lang, lang_records in sorted(by_lang.items()):
        out = args.output_dir / f"benchmark_{lang}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in lang_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {lang:10s}: {len(lang_records):4d} records → {out.name}")


if __name__ == "__main__":
    main()