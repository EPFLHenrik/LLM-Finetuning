"""multilingual/train/build_sft_data.py

Builds SFT JSONL files from all loaders.

Output layout:
    data/sft/per_dataset/<dataset>_<language>_<split>.jsonl  -- full, no caps
    data/sft/mixed/train_mixed.jsonl                         -- capped + shuffled
    data/sft/mixed/val_mixed.jsonl                           -- capped + shuffled
    data/sft/build_summary.csv

Usage:
    python -m multilingual.train.build_sft_data --debug
    python -m multilingual.train.build_sft_data
    python -m multilingual.train.build_sft_data --max_per_language 500 --max_per_dataset 2000
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

from multilingual.config import REPO, SYSTEM_PROMPT, TARGET_LANGUAGES
from multilingual.loaders.ceval_loader import load_ceval_examples
from multilingual.loaders.include_loader import load_include_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples
from multilingual.loaders.mmmlu_loader import load_mmmlu_examples

DEFAULT_MAX_PER_LANGUAGE = 1000
DEFAULT_MAX_PER_DATASET  = 4000

DEBUG_MAX_TRAIN = 20
DEBUG_MAX_VAL   = 10

SHUFFLE_SEED = 42

SFT_DATA_DIR = REPO / "multilingual" / "data" / "sft"

# Dataset → applicable languages (None = all TARGET_LANGUAGES)
DATASET_LANGUAGES = {
    "include": None,
    "mgsm":    None,
    "mmmlu":   ["spanish", "hindi", "italian", "chinese"],
    "milu":    ["hindi"],
    "ceval":   ["chinese"],
}

def hash_source_id(ex: Dict[str, Any]) -> str:
    raw = "\n".join([
        str(ex["dataset"]),
        str(ex["language"]),
        str(ex["task_type"]),
        str(ex["prompt"]).strip(),
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{ex['dataset']}_{ex['language']}_{digest}"


def _make_assistant_message(ex: Dict[str, Any]) -> str:
    answer = str(ex["answer"]).strip()
    if ex["task_type"] == "mcq":
        answer = answer.upper()
    return f"\\boxed{{{answer}}}"


def to_sft_record(ex: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    """Convert a raw loader example to an SFT record."""
    return {
        "id":        f"{ex['dataset']}_{split}_{ex['language']}_{idx}",
        "source_id": hash_source_id(ex),
        "loader_id": ex["id"],
        "dataset":   ex["dataset"],
        "language":  ex["language"],
        "split":     split,
        "task_type": ex["task_type"],
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ex["prompt"]},
            {"role": "assistant", "content": _make_assistant_message(ex)},
        ],
        "answer":   ex["answer"],
        "metadata": ex.get("metadata", {}),
    }


def _dedupe_by_source_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate prompts within one split, preserving first occurrence."""
    seen = set()
    deduped = []

    for r in records:
        sid = r["source_id"]
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(r)

    return deduped


def _remove_train_val_overlap(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove train records whose prompt also appears in validation."""
    val_ids = {r["source_id"] for r in val_records}
    return [r for r in train_records if r["source_id"] not in val_ids]

def _write_jsonl(records: List[Dict], path: Path, overwrite: bool) -> bool:
    """Write records to path. Returns False and skips if file exists and not overwrite."""
    if path.exists() and not overwrite:
        print(f"  [SKIP] exists: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>5} records → {path}")
    return True


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_examples(
    dataset: str, language: str, split: str,
    n: Optional[int], hf_token: str,
) -> List[Dict[str, Any]]:
    if dataset == "include":
        return load_include_examples(language, n=n, split=split)
    if dataset == "mgsm":
        return load_mgsm_examples(language, n=n, split=split)
    if dataset == "mmmlu":
        return load_mmmlu_examples(language, n=n, split=split)
    if dataset == "milu":
        return load_milu_examples(n=n, split=split, hf_token=hf_token)
    if dataset == "ceval":
        return load_ceval_examples(n=n, split=split)
    raise ValueError(f"Unknown dataset: {dataset}")


def _iter_pairs():
    for dataset, langs in DATASET_LANGUAGES.items():
        languages = langs if langs is not None else list(TARGET_LANGUAGES.keys())
        for language in languages:
            yield dataset, language


def build(
    max_per_language: int,
    max_per_dataset: int,
    debug: bool,
    overwrite: bool,
    hf_token: str,
) -> None:
    per_dir   = SFT_DATA_DIR / "per_dataset"
    mixed_dir = SFT_DATA_DIR / "mixed"
    rng = random.Random(SHUFFLE_SEED)
    summary_rows = []

    # Step 1: per-dataset/language files
    for dataset, language in _iter_pairs():
        split_records: Dict[str, List[Dict[str, Any]]] = {}

        for split in ("train", "val"):
            n_cap = (DEBUG_MAX_TRAIN if split == "train" else DEBUG_MAX_VAL) if debug else None

            print(f"Loading {dataset}/{language}/{split} ...")
            try:
                examples = _load_examples(dataset, language, split, n_cap, hf_token)
            except Exception as e:
                print(f"  [SKIP] {e}")
                split_records[split] = []
                continue

            if not examples:
                print(f"  [SKIP] empty")
                split_records[split] = []
                continue

            records = [to_sft_record(ex, split, i) for i, ex in enumerate(examples)]

            before = len(records)
            records = _dedupe_by_source_id(records)
            removed = before - len(records)
            if removed:
                print(f"  Removed {removed} duplicate source_ids from {dataset}/{language}/{split}")

            split_records[split] = records

        # remove train examples that duplicate validation prompts.
        if split_records.get("train") and split_records.get("val"):
            before = len(split_records["train"])
            split_records["train"] = _remove_train_val_overlap(
                split_records["train"],
                split_records["val"],
            )
            removed = before - len(split_records["train"])
            if removed:
                print(f"  Removed {removed} train/val overlaps from {dataset}/{language}")

        # write cleaned train/val files.
        for split in ("train", "val"):
            records = split_records.get(split, [])
            if not records:
                continue

            # re-index ids after deduplication/filtering.
            for i, r in enumerate(records):
                r["id"] = f"{r['dataset']}_{split}_{r['language']}_{i}"
                r["split"] = split

            out_path = per_dir / f"{dataset}_{language}_{split}.jsonl"
            _write_jsonl(records, out_path, overwrite)

            summary_rows.append({
                "dataset": dataset,
                "language": language,
                "split": split,
                "n": len(records),
                "path": str(out_path),
            })

    # Step 2: mixed files sampled from per-dataset files
    for split in ("train", "val"):
        mixed: List[Dict] = []

        for dataset, langs in DATASET_LANGUAGES.items():
            languages = langs if langs is not None else list(TARGET_LANGUAGES.keys())

            # load available language pools for this dataset/split
            lang_pools: Dict[str, List[Dict]] = {}
            for language in languages:
                path = per_dir / f"{dataset}_{language}_{split}.jsonl"
                if not path.exists():
                    continue

                lang_records = _load_jsonl(path)
                if not lang_records:
                    continue

                rng.shuffle(lang_records)
                lang_pools[language] = lang_records

            if not lang_pools:
                continue

            # in debug mode, files are already small; keep all examples.
            if debug:
                dataset_pool = []
                for records in lang_pools.values():
                    dataset_pool.extend(records)
                rng.shuffle(dataset_pool)
                mixed.extend(dataset_pool)
                continue

            # Balanced per-language quota inside this dataset.
            available_languages = list(lang_pools.keys())
            n_langs = len(available_languages)
            per_lang_quota = max(1, max_per_dataset // n_langs)

            selected_by_lang: Dict[str, List[Dict]] = {}
            leftovers_by_lang: Dict[str, List[Dict]] = {}

            for language, records in lang_pools.items():
                quota = min(max_per_language, per_lang_quota)
                selected_by_lang[language] = records[:quota]
                leftovers_by_lang[language] = records[quota:max_per_language]

            dataset_pool: List[Dict] = []
            for records in selected_by_lang.values():
                dataset_pool.extend(records)

            # redistribute remaining dataset capacity to languages with leftovers
            remaining_capacity = max_per_dataset - len(dataset_pool)
            if remaining_capacity > 0:
                leftovers: List[Dict] = []
                for records in leftovers_by_lang.values():
                    leftovers.extend(records)

                rng.shuffle(leftovers)
                dataset_pool.extend(leftovers[:remaining_capacity])

            rng.shuffle(dataset_pool)
            dataset_pool = dataset_pool[:max_per_dataset]
            mixed.extend(dataset_pool)

        rng.shuffle(mixed)
        mixed_path = mixed_dir / f"{split}_mixed.jsonl"
        _write_jsonl(mixed, mixed_path, overwrite)
        summary_rows.append({
            "dataset": "mixed", "language": "all",
            "split": split, "n": len(mixed), "path": str(mixed_path),
        })

    # Step 3: summary CSV
    summary_path = SFT_DATA_DIR / "build_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "language", "split", "n", "path"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_language", type=int, default=DEFAULT_MAX_PER_LANGUAGE)
    parser.add_argument("--max_per_dataset",  type=int, default=DEFAULT_MAX_PER_DATASET)
    parser.add_argument("--debug", action="store_true",
        help=f"Small build: {DEBUG_MAX_TRAIN} train / {DEBUG_MAX_VAL} val per dataset/language.")
    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite existing output files.")
    return parser.parse_args()


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"Debug: {args.debug}  Overwrite: {args.overwrite}")
    print(f"Caps:  max_per_language={args.max_per_language}, max_per_dataset={args.max_per_dataset}")
    print(f"Output: {SFT_DATA_DIR}\n")
    build(
        max_per_language=args.max_per_language,
        max_per_dataset=args.max_per_dataset,
        debug=args.debug,
        overwrite=args.overwrite,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()