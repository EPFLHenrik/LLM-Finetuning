#!/usr/bin/env python3
"""
Count distilled examples per language and dataset.
Usage:
    python multilingual/data/distilled/count.py multilingual/data/distilled/v2_fallback/sft_1
"""

from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Folder containing *_distilled.jsonl files")
    return parser.parse_args()


def main():
    args = parse_args()
    folder = args.folder
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    files = sorted(folder.glob("*_distilled.jsonl"))

    if not files:
        print(f"No *_distilled.jsonl files found in {folder}")
        return

    language_counts  = Counter()
    dataset_counts   = Counter()
    file_counts      = Counter()
    # dataset -> language -> count
    ds_lang_counts: dict[str, Counter] = defaultdict(Counter)
    # language -> dataset -> count
    lang_ds_counts: dict[str, Counter] = defaultdict(Counter)
    total = 0

    for path in files:
        n_file = 0
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON: {path}:{line_no}")
                    continue

                language = ex.get("language") or ex.get("metadata", {}).get("language", "unknown")
                language = str(language).strip().lower() or "unknown"
                dataset  = str(ex.get("dataset", "unknown")).strip().lower()

                language_counts[language] += 1
                dataset_counts[dataset]   += 1
                file_counts[path.name]    += 1
                ds_lang_counts[dataset][language] += 1
                lang_ds_counts[language][dataset] += 1
                n_file += 1
                total  += 1

        print(f"{path.name}: {n_file}")

    print(f"\n{'='*50}")
    print(f"TOTAL: {total}")

    print(f"\n{'='*50}")
    print("Per language:")
    for lang, count in language_counts.most_common():
        print(f"  {lang:<14} {count:>6}")

    print(f"\n{'='*50}")
    print("Per dataset:")
    for ds, count in dataset_counts.most_common():
        print(f"  {ds:<30} {count:>6}")

    print(f"\n{'='*50}")
    print("Per dataset → language:")
    for ds, lang_ctr in sorted(ds_lang_counts.items()):
        print(f"  {ds}:")
        for lang, count in lang_ctr.most_common():
            print(f"    {lang:<14} {count:>6}")

    print(f"\n{'='*50}")
    print("Per language → dataset:")
    for lang, ds_ctr in sorted(lang_ds_counts.items()):
        print(f"  {lang}:")
        for ds, count in ds_ctr.most_common():
            print(f"    {ds:<30} {count:>6}")


if __name__ == "__main__":
    main()