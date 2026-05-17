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
    python -m multilingual.train.build_sft_data --max_per_language 500 --max_per_dataset 2000 --overwrite
"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from multilingual.config import REPO, SYSTEM_PROMPT, TARGET_LANGUAGES
from multilingual.loaders.belebele_loader import load_belebele_examples
from multilingual.loaders.ceval_loader import load_ceval_examples
from multilingual.loaders.cultsportqa_loader import load_cultsportqa_examples
from multilingual.loaders.exams_loader import load_exams_examples
from multilingual.loaders.head_qa_loader import load_head_qa_examples
from multilingual.loaders.include_loader import load_include_examples
from multilingual.loaders.mera_loader import load_mera_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples
from multilingual.loaders.mmmlu_loader import load_mmmlu_examples
from multilingual.loaders.xcopa_loader import load_xcopa_examples
from multilingual.loaders.global_mmlu_loader import load_global_mmlu_examples
from multilingual.loaders.global_piqa_loader import load_global_piqa_examples



# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_PER_LANGUAGE = 1500
DEFAULT_MAX_PER_DATASET  = 7000
DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL = 2500
# Target examples per language in the final mixed file.
# Set to None to disable final language balancing.
DEFAULT_MAX_PER_LANGUAGE_MIXED = 6000
DEBUG_MAX_TRAIN = 20
DEBUG_MAX_VAL   = 10
SHUFFLE_SEED    = 42

SFT_DATA_DIR = REPO / "multilingual" / "data" / "sft"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry:
#   languages:          list of language keys, or None for all TARGET_LANGUAGES
#   max_per_language:   override global cap (None = use global)
#   max_per_dataset:    override global cap (None = use global)
#   stratify_by:        column in metadata to stratify sampling by (None = no stratification)
#   subject_caps:       {subject: cap} overrides when stratify_by="subject"
#                       None value means "keep all". Missing subjects use even distribution.
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "include": {
        "languages": None,  # all 5 target languages
        "language_caps": {
            "spanish": 2500,
            "italian": 2500,
            "russian": 2500,
            "chinese": 2500,
            "hindi":   2500,
        },
    },
    "mgsm": {
        "languages": None,
    },
    # we added global-MMLU instead
    "mmmlu": {
        "languages": ["spanish", "hindi", "italian", "chinese"],
        "max_per_dataset": 0,  # disabled, replaced by global_mmlu
    },
    "milu": {
        "languages": ["hindi"],
        "max_per_language": 3000,
    },
    "ceval": {
        "languages": ["chinese"],
        "max_per_dataset": 3000,
        "language_caps": {
            "chinese": 3000,
        }

    },
    "cultsportqa": {
        "languages": ["chinese", "hindi"],
        "max_per_language": 8000,
        "max_per_dataset":  8000,
    },
    "belebele": {
        "languages": None,
    },
    "xcopa": {
        "languages": ["italian", "chinese"],
    },
    "exams": {
        "languages": ["italian", "spanish"],
        # small dataset — keep everything, no cap
        "max_per_language": None,
        "max_per_dataset":  None,
    },
    "head_qa": {
        "languages": ["spanish", "italian"],
        # italian is machine-translated — cap lower than Spanish
        "language_caps": {
            "spanish": 2000,
            "italian": 1000,
        },
    },
    "mera": {
        "languages": ["russian"],
        # single language, include as much as possible
        "max_per_language": 8000,
        "max_per_dataset":  8000,
        "stratify_by": "subset",
        "subject_caps": {
            "use":          1000,   # numeric, cap it — already large
            "ruopenbookqa": 1000,   # cap the largest subset
            "rwsd":         None,  # keep all 810
            "rcb":          None,  # keep all 658
            "mathlogicqa":  None,  # keep all 680
            "parus":        None,  # keep all 500
            "mamuramu":     None,  # keep all 285
            "ruworldtree":  None,  # keep all 115
        },
    },
    "global_mmlu": {
        "languages": None,  # all 5
        "language_caps": {
            "italian": 3000,
            "russian": 3000,
            "spanish": 3000,
            "hindi":   1000,
            "chinese": 1000,
        },
    },
    "global_piqa": {
        "languages": None,
        "max_per_language": None,
        "max_per_dataset":  None,
    },
}


def _dataset_cfg(dataset: str) -> Dict[str, Any]:
    return DATASET_REGISTRY.get(dataset, {})


def _languages(dataset: str) -> List[str]:
    cfg = _dataset_cfg(dataset)
    langs = cfg.get("languages")
    return langs if langs is not None else list(TARGET_LANGUAGES.keys())


def _max_per_language(dataset: str, language: str, global_cap: int) -> int:
    cfg = _dataset_cfg(dataset)
    # Per-language-per-dataset override takes highest priority
    lang_caps = cfg.get("language_caps", {})
    if language in lang_caps:
        v = lang_caps[language]
        return v if v is not None else 10_000_000
    # Dataset-level override
    v = cfg.get("max_per_language", global_cap)
    return v if v is not None else 10_000_000


def _max_per_dataset(dataset: str, global_cap: int) -> int:
    cfg = _dataset_cfg(dataset)
    v = cfg.get("max_per_dataset", global_cap)
    return v if v is not None else 10_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_source_id(ex: Dict[str, Any]) -> str:
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


def _to_sft_record(ex: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    return {
        "id":        f"{ex['dataset']}_{split}_{ex['language']}_{idx}",
        "source_id": _hash_source_id(ex),
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

def _dedupe(records: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in records:
        if r["source_id"] not in seen:
            seen.add(r["source_id"])
            out.append(r)
    return out

def _remove_overlap(train: List[Dict], val: List[Dict]) -> List[Dict]:
    val_ids = {r["source_id"] for r in val}
    return [r for r in train if r["source_id"] not in val_ids]


def _write_jsonl(records: List[Dict], path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"  [SKIP] exists: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>5} records → {path.name}")
    return True


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _reindex(records: List[Dict], split: str) -> List[Dict]:
    for i, r in enumerate(records):
        r["id"] = f"{r['dataset']}_{split}_{r['language']}_{i}"
        r["split"] = split
    return records

# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------

def _load_examples(
    dataset: str, language: str, split: str,
    n: Optional[int], hf_token: str,
) -> List[Dict[str, Any]]:
    kw = dict(language=language, n=n, split=split)
    if dataset == "include":    return load_include_examples(**kw)
    if dataset == "mgsm":       return load_mgsm_examples(**kw)
    if dataset == "mmmlu":      return load_mmmlu_examples(**kw)
    if dataset == "milu":       return load_milu_examples(n=n, split=split, hf_token=hf_token)
    if dataset == "ceval":      return load_ceval_examples(n=n, split=split)
    if dataset == "cultsportqa":return load_cultsportqa_examples(**kw)
    if dataset == "belebele":   return load_belebele_examples(**kw)
    if dataset == "xcopa":      return load_xcopa_examples(**kw)
    if dataset == "exams":      return load_exams_examples(**kw)
    if dataset == "head_qa":    return load_head_qa_examples(**kw)
    if dataset == "mera":       return load_mera_examples(**kw)
    if dataset == "global_mmlu": return load_global_mmlu_examples(**kw)
    if dataset == "global_piqa": return load_global_piqa_examples(**kw)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stratified_sample(
    records: List[Dict],
    stratify_by: str,
    subject_caps: Dict[str, Optional[int]],
    global_cap: int,
    rng: random.Random,
) -> List[Dict]:
    """
    Sample records stratified by a metadata field.

    subject_caps: {subject: cap_or_None}. None means keep all.
    Subjects not in subject_caps get an even share of the remaining global_cap.
    """
    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        key = str(r.get("metadata", {}).get(stratify_by, "unknown"))
        by_group[key].append(r)

    # Shuffle within each group
    for g in by_group.values():
        rng.shuffle(g)

    result = []
    # First pass: apply explicit caps
    uncapped_groups = []
    used = 0
    for group, items in by_group.items():
        if group in subject_caps:
            cap = subject_caps[group]
            take = items if cap is None else items[:cap]
            result.extend(take)
            used += len(take)
        else:
            uncapped_groups.append((group, items))

    # Second pass: distribute remaining capacity evenly among uncapped groups
    remaining = max(0, global_cap - used)
    if uncapped_groups and remaining > 0:
        per_group = max(1, remaining // len(uncapped_groups))
        for _, items in uncapped_groups:
            result.extend(items[:per_group])

    rng.shuffle(result)
    return result[:global_cap]


# ---------------------------------------------------------------------------
# Mixed file builder
# ---------------------------------------------------------------------------

def _build_mixed(
    per_dir: Path,
    split: str,
    global_max_per_language: int,
    global_max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    rng: random.Random,
    debug: bool,
) -> List[Dict]:
    mixed: List[Dict] = []

    for dataset, cfg in DATASET_REGISTRY.items():
        languages = _languages(dataset)
        lang_pools: Dict[str, List[Dict]] = {}

        for language in languages:
            path = per_dir / f"{dataset}_{language}_{split}.jsonl"
            if not path.exists():
                continue
            records = _load_jsonl(path)
            if not records:
                continue
            rng.shuffle(records)
            lang_pools[language] = records

        if not lang_pools:
            continue

        if debug:
            for recs in lang_pools.values():
                mixed.extend(recs)
            continue

        # per-language caps
        cfg = _dataset_cfg(dataset)
        stratify_by  = cfg.get("stratify_by")
        subject_caps = cfg.get("subject_caps", {})

        dataset_pool: List[Dict] = []
        for language, records in lang_pools.items():
            cap = _max_per_language(dataset, language, global_max_per_language)
            if stratify_by:
                sampled = _stratified_sample(
                    records, stratify_by, subject_caps, cap, rng
                )
            else:
                sampled = records[:cap]
            dataset_pool.extend(sampled)

        # dataset cap
        ds_cap = _max_per_dataset(dataset, global_max_per_dataset)
        rng.shuffle(dataset_pool)
        mixed.extend(dataset_pool[:ds_cap])

    if not debug and max_per_language_mixed is not None:
        # Final language balancing: cap each language's contribution
        by_lang: Dict[str, List[Dict]] = defaultdict(list)
        for r in mixed:
            by_lang[r["language"]].append(r)

        balanced: List[Dict] = []
        for lang, records in by_lang.items():
            rng.shuffle(records)
            balanced.extend(records[:max_per_language_mixed])
        mixed = balanced

    rng.shuffle(mixed)
    return mixed


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(
    max_per_language: int,
    max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    debug: bool,
    overwrite: bool,
    hf_token: str,
) -> None:
    per_dir   = SFT_DATA_DIR / "per_dataset"
    mixed_dir = SFT_DATA_DIR / "mixed"
    rng       = random.Random(SHUFFLE_SEED)
    summary_rows: List[Dict] = []

    # step 1: per-dataset/language files (full, no caps)
    for dataset, cfg in DATASET_REGISTRY.items():
        for language in _languages(dataset):
            split_records: Dict[str, List[Dict]] = {}

            for split in ("train", "val"):
                if debug:
                    n_cap = DEBUG_MAX_TRAIN if split == "train" else DEBUG_MAX_VAL
                else:
                    n_cap = None  # load full data; caps applied in _build_mixed
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

                records = [_to_sft_record(ex, split, i) for i, ex in enumerate(examples)]

                before = len(records)
                records = _dedupe(records)
                if len(records) < before:
                    print(f"  Deduped {before - len(records)} rows")

                split_records[split] = records

            # remove train/val overlap
            if split_records.get("train") and split_records.get("val"):
                before = len(split_records["train"])
                split_records["train"] = _remove_overlap(
                    split_records["train"], split_records["val"]
                )
                removed = before - len(split_records["train"])
                if removed:
                    print(f"  Removed {removed} train/val overlaps from {dataset}/{language}")

            for split in ("train", "val"):
                records = split_records.get(split, [])
                if not records:
                    continue
                records = _reindex(records, split)
                out_path = per_dir / f"{dataset}_{language}_{split}.jsonl"
                if _write_jsonl(records, out_path, overwrite):
                    summary_rows.append({
                        "dataset": dataset, "language": language,
                        "split": split, "n": len(records), "path": str(out_path),
                    })

    # step 2: mixed files
    for split in ("train", "val"):
        cap = max_per_language_mixed if split == "train" else DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL
        mixed = _build_mixed(
            per_dir=per_dir,
            split=split,
            global_max_per_language=max_per_language,
            global_max_per_dataset=max_per_dataset,
            max_per_language_mixed=cap,
            rng=rng,
            debug=debug,
        )
        mixed_path = mixed_dir / f"{split}_mixed.jsonl"
        if _write_jsonl(mixed, mixed_path, overwrite):
            summary_rows.append({
                "dataset": "mixed", "language": "all",
                "split": split, "n": len(mixed), "path": str(mixed_path),
            })

    # step 3: summary CSV
    summary_path = SFT_DATA_DIR / "build_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "language", "split", "n", "path"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_language",       type=int, default=DEFAULT_MAX_PER_LANGUAGE)
    parser.add_argument("--max_per_dataset",        type=int, default=DEFAULT_MAX_PER_DATASET)
    parser.add_argument("--max_per_language_mixed", type=int, default=DEFAULT_MAX_PER_LANGUAGE_MIXED,
        help="Final per-language cap in the mixed file. Set 0 to disable.")
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    max_mixed = args.max_per_language_mixed if args.max_per_language_mixed > 0 else None
    hf_token  = os.environ.get("HF_TOKEN", "")

    print(f"Debug:     {args.debug}  Overwrite: {args.overwrite}")
    print(f"Caps:      max_per_language={args.max_per_language}, "
          f"max_per_dataset={args.max_per_dataset}, "
          f"max_per_language_mixed={max_mixed}")
    print(f"Output:    {SFT_DATA_DIR}\n")

    build(
        max_per_language=args.max_per_language,
        max_per_dataset=args.max_per_dataset,
        max_per_language_mixed=max_mixed,
        debug=args.debug,
        overwrite=args.overwrite,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()