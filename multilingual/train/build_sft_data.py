"""multilingual/train/build_sft_data.py

Builds SFT JSONL files from all loaders.

Output layout:
    <output_dir>/per_dataset/<dataset>_<language>_<split>.jsonl
    <output_dir>/mixed/train_mixed.jsonl
    <output_dir>/mixed/val_mixed.jsonl
    <output_dir>/mixed/test_mixed.jsonl   (if any dataset has test data)
    <output_dir>/build_summary.csv

Output dir:
    Default: data/sft/
    Override: set "output_name" in JSON config → data/sft/<output_name>/

Dataset modes:
    as_is   (default): load each split from HF directly.
    recarve:           merge all non-train HF splits, recarve into val/test
                       using split_caps sizes (default 50/50).
                       Datasets with mode=recarve never produce a train split.

Usage:
    python -m multilingual.train.build_sft_data
    python -m multilingual.train.build_sft_data --config configs/v7.json --overwrite
    python -m multilingual.train.build_sft_data --debug
"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from multilingual.config import REPO, SYSTEM_PROMPT, TARGET_LANGUAGES
from multilingual.loaders.belebele_loader import load_belebele_examples
from multilingual.loaders.ceval_loader import load_ceval_examples
from multilingual.loaders.cultsportqa_loader import load_cultsportqa_examples
from multilingual.loaders.exams_loader import load_exams_examples
from multilingual.loaders.head_qa_loader import load_head_qa_examples
from multilingual.loaders.include_loader import load_include_recarved, load_include_examples
from multilingual.loaders.mera_loader import load_mera_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples
from multilingual.loaders.mmmlu_loader import load_mmmlu_examples
from multilingual.loaders.xcopa_loader import load_xcopa_examples
from multilingual.loaders.global_mmlu_loader import load_global_mmlu_examples
from multilingual.loaders.global_piqa_loader import load_global_piqa_recarved, load_global_piqa_examples

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_PER_LANGUAGE           = 1000
DEFAULT_MAX_PER_DATASET            = 5000
DEFAULT_MAX_PER_LANGUAGE_MIXED     = 2500
DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL = 1000
DEFAULT_SFT_BASE_DIR = REPO / "multilingual" / "data" / "sft"
DEBUG_MAX            = 20
SHUFFLE_SEED         = 42
ALL_SPLITS           = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "include": {
        "languages":  None,
        "mode":       "recarve",   # merge HF validation+test → recarve
        "split_caps": {"train": 0, "val": None, "test": None},  # None = 50/50
    },
    "global_piqa": {
        "languages":  None,
        "mode":       "recarve",   # merge HF test → recarve
        "split_caps": {"train": 0, "val": None, "test": None},  # None = 50/50
        "max_per_dataset": 0,      # excluded from mixed train
    },
    "mgsm": {
        "languages":        None,
        "max_per_language": 50,
        "max_per_dataset":  250,
        "split_caps":       {"val": 100},
    },
    "mmmlu": {
        "languages":       ["spanish", "hindi", "italian", "chinese"],
        "max_per_dataset": 0,
    },
    "milu": {
        "languages":        ["hindi"],
        "max_per_language": 2000,
        "split_caps":       {"val": 300},
    },
    "ceval": {
        "languages":    ["chinese"],
        "language_caps": {"chinese": 2000},
        "split_caps":   {"val": 300},
    },
    "cultsportqa": {
        "languages":        ["chinese", "hindi"],
        "max_per_language": 1000,
        "max_per_dataset":  2000,
        "split_caps":       {"val": 400},
    },
    "belebele": {
        "languages":        None,
        "max_per_language": 300,
        "max_per_dataset":  1500,
        "split_caps":       {"val": 100},
    },
    "xcopa": {
        "languages":       ["italian", "chinese"],
        "max_per_dataset": 0,
    },
    "exams": {
        "languages":  ["italian", "spanish"],
        "split_caps": {"val": 300},
    },
    "head_qa": {
        "languages":     ["spanish", "italian"],
        "language_caps": {"spanish": 1000, "italian": 400},
        "split_caps":    {"val": 300},
    },
    "mera": {
        "languages":        ["russian"],
        "max_per_language": 2000,
        "max_per_dataset":  2000,
        "stratify_by":      "subset",
        "subject_caps": {
            "mamuramu":     285,
            "ruopenbookqa": 600,
            "ruworldtree":  115,
            "mathlogicqa":  150,
            "parus":          0,
            "use":            0,
            "rcb":            0,
            "rwsd":           0,
        },
        "split_caps": {"val": 200},
    },
    "global_mmlu": {
        "languages":     None,
        "language_caps": {
            "russian": 2200,
            "chinese": 2200,
            "hindi":    600,
            "italian":  300,
            "spanish":  300,
        },
        "split_caps": {"val": 600},
    },
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _dataset_cfg(dataset: str) -> Dict[str, Any]:
    return DATASET_REGISTRY.get(dataset, {})


def _languages(dataset: str) -> List[str]:
    cfg   = _dataset_cfg(dataset)
    langs = cfg.get("languages")
    return langs if langs is not None else list(TARGET_LANGUAGES.keys())


def _max_per_language(dataset: str, language: str, global_cap: int) -> int:
    cfg       = _dataset_cfg(dataset)
    lang_caps = cfg.get("language_caps", {})
    if language in lang_caps:
        v = lang_caps[language]
        return v if v is not None else 10_000_000
    v = cfg.get("max_per_language", global_cap)
    return v if v is not None else 10_000_000


def _split_cap(dataset: str, split: str, global_cap: int) -> int:
    """Return the effective cap for dataset+split. 0 = disabled."""
    cfg        = _dataset_cfg(dataset)
    split_caps = cfg.get("split_caps", {})
    if split in split_caps:
        v = split_caps[split]
        return v if v is not None else 10_000_000
    if split == "test":
        return 0  # test only produced if explicitly set
    v = cfg.get("max_per_dataset", global_cap)
    return v if v is not None else 10_000_000


def _split_enabled(dataset: str, split: str) -> bool:
    return _split_cap(dataset, split, 10_000_000) != 0


def _is_recarve(dataset: str) -> bool:
    return _dataset_cfg(dataset).get("mode") == "recarve"

# ---------------------------------------------------------------------------
# SFT record helpers
# ---------------------------------------------------------------------------

def _hash_source_id(ex: Dict[str, Any]) -> str:
    raw = "\n".join([
        str(ex["dataset"]), str(ex["language"]),
        str(ex["task_type"]), str(ex["prompt"]).strip(),
    ])
    return f"{ex['dataset']}_{ex['language']}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _make_assistant_message(ex: Dict[str, Any]) -> str:
    answer = str(ex["answer"]).strip()
    if ex["task_type"] == "mcq":
        answer = answer.upper()
    boxed = f"\\boxed{{{answer}}}"
    cot = ex.get("cot") or ex.get("rationale") or ex.get("reasoning")
    if cot:
        cot = str(cot).strip()
        return f"<think>\n{cot}\n</think>\n\n{boxed}"
    return boxed


def _to_sft_record(
    ex: Dict[str, Any], split: str, idx: int,
) -> Dict[str, Any]:
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


def _to_records(
    examples: List[Dict], split: str,
) -> List[Dict]:
    return [_to_sft_record(ex, split, i) for i, ex in enumerate(examples)]


def _dedupe(records: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in records:
        if r["source_id"] not in seen:
            seen.add(r["source_id"])
            out.append(r)
    return out


def _remove_train_overlap(train: List[Dict], others: List[List[Dict]]) -> List[Dict]:
    other_ids = {r["source_id"] for split in others for r in split}
    return [r for r in train if r["source_id"] not in other_ids]


def _reindex(records: List[Dict], split: str) -> List[Dict]:
    for i, r in enumerate(records):
        r["id"]    = f"{r['dataset']}_{split}_{r['language']}_{i}"
        r["split"] = split
    return records


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

# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------

def _load_examples(
    dataset: str, language: str, split: str,
    n: Optional[int], hf_token: str,
) -> List[Dict[str, Any]]:
    """Load examples for a single dataset/language/split (as_is mode only)."""
    mode = str(_dataset_cfg(dataset).get("mode", "as_is"))

    if dataset == "include":
        return load_include_examples(language=language, n=n, split=split, mode=mode)
    if dataset == "global_piqa":
        return load_global_piqa_examples(language=language, n=n, split=split, mode=mode)
    if dataset == "mgsm":
        return load_mgsm_examples(language=language, n=n, split=split)
    if dataset == "mmmlu":
        return load_mmmlu_examples(language=language, n=n, split=split)
    if dataset == "milu":
        return load_milu_examples(n=n, split=split, hf_token=hf_token)
    if dataset == "ceval":
        return load_ceval_examples(n=n, split=split)
    if dataset == "cultsportqa":
        return load_cultsportqa_examples(language=language, n=n, split=split)
    if dataset == "belebele":
        return load_belebele_examples(language=language, n=n, split=split)
    if dataset == "xcopa":
        return load_xcopa_examples(language=language, n=n, split=split)
    if dataset == "exams":
        return load_exams_examples(language=language, n=n, split=split)
    if dataset == "head_qa":
        return load_head_qa_examples(language=language, n=n, split=split)
    if dataset == "mera":
        return load_mera_examples(language=language, n=n, split=split)
    if dataset == "global_mmlu":
        return load_global_mmlu_examples(language=language, n=n, split=split)
    raise ValueError(f"Unknown dataset: {dataset}")


def _load_recarved(
    dataset: str, language: str,
    n_val: Optional[int], n_test: Optional[int],
) -> Dict[str, List[Dict]]:
    """Load and recarve a dataset into {split: records}. Single HF download."""
    if dataset == "include":
        val_ex, test_ex = load_include_recarved(language, n_val=n_val, n_test=n_test)
    elif dataset == "global_piqa":
        val_ex, test_ex = load_global_piqa_recarved(language, n_val=n_val, n_test=n_test)
    else:
        raise ValueError(f"Recarve not implemented for dataset: {dataset}")

    result = {}
    if val_ex:
        result["val"]  = _dedupe(_to_records(val_ex,  "val"))
    if test_ex:
        result["test"] = _dedupe(_to_records(test_ex, "test"))
    return result

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
    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        key = str(r.get("metadata", {}).get(stratify_by, "unknown"))
        by_group[key].append(r)

    for g in by_group.values():
        rng.shuffle(g)

    result, used, uncapped = [], 0, []
    for group, items in by_group.items():
        if group in subject_caps:
            cap = subject_caps[group]
            if cap == 0:
                continue
            take = items if cap is None else items[:cap]
            result.extend(take)
            used += len(take)
        else:
            uncapped.append((group, items))

    remaining = max(0, global_cap - used)
    if uncapped and remaining > 0:
        per_group = max(1, remaining // len(uncapped))
        for _, items in uncapped:
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

    for dataset in DATASET_REGISTRY:
        if not _split_enabled(dataset, split):
            continue

        cfg          = _dataset_cfg(dataset)
        stratify_by  = cfg.get("stratify_by")
        subject_caps = cfg.get("subject_caps", {})
        dataset_pool: List[Dict] = []

        for language in _languages(dataset):
            path = per_dir / f"{dataset}_{language}_{split}.jsonl"
            if not path.exists():
                continue
            records = _load_jsonl(path)
            if not records:
                continue
            rng.shuffle(records)

            if debug:
                dataset_pool.extend(records)
                continue

            cap = _max_per_language(dataset, language, global_max_per_language)
            if stratify_by:
                sampled = _stratified_sample(records, stratify_by, subject_caps, cap, rng)
            else:
                sampled = records[:cap]
            dataset_pool.extend(sampled)

        if not dataset_pool:
            continue

        if not debug:
            ds_cap = _split_cap(dataset, split, global_max_per_dataset)
            rng.shuffle(dataset_pool)
            dataset_pool = dataset_pool[:ds_cap]

        mixed.extend(dataset_pool)

    if not debug and max_per_language_mixed is not None:
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
    output_dir: Path,
    max_per_language: int,
    max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    debug: bool,
    overwrite: bool,
    hf_token: str,
) -> None:
    per_dir   = output_dir / "per_dataset"
    mixed_dir = output_dir / "mixed"
    rng       = random.Random(SHUFFLE_SEED)
    summary_rows: List[Dict] = []

    # Step 1: per-dataset/language files
    for dataset in DATASET_REGISTRY:
        for language in _languages(dataset):
            print(f"\n── {dataset}/{language} ──")

            if _is_recarve(dataset):
                # Single HF download → both val and test
                n_val  = _split_cap(dataset, "val",  max_per_dataset)
                n_test = _split_cap(dataset, "test", max_per_dataset)
                n_val  = None if n_val  >= 10_000_000 else n_val
                n_test = None if n_test >= 10_000_000 else n_test

                try:
                    split_records = _load_recarved(dataset, language, n_val, n_test)
                except Exception as e:
                    print(f"  [SKIP recarve] {e}")
                    continue
            else:
                # Normal per-split loading
                active = [s for s in ALL_SPLITS if _split_enabled(dataset, s)]
                raw: Dict[str, List[Dict]] = {}

                for split in active:
                    n_cap = DEBUG_MAX if debug else None
                    print(f"  Loading {split} ...")
                    try:
                        examples = _load_examples(dataset, language, split, n_cap, hf_token)
                    except Exception as e:
                        print(f"  [SKIP] {e}")
                        raw[split] = []
                        continue
                    if not examples:
                        print(f"  [SKIP] empty")
                        raw[split] = []
                        continue
                    records = _dedupe(_to_records(examples, split))
                    raw[split] = records

                # Remove train overlap with val + test
                if raw.get("train"):
                    others = [v for k, v in raw.items() if k != "train" and v]
                    before = len(raw["train"])
                    raw["train"] = _remove_train_overlap(raw["train"], others)
                    removed = before - len(raw["train"])
                    if removed:
                        print(f"  Removed {removed} train/eval overlaps")

                split_records = {k: v for k, v in raw.items() if v}

            # Write per-dataset files
            for split, records in split_records.items():
                records = _reindex(records, split)
                out_path = per_dir / f"{dataset}_{language}_{split}.jsonl"
                if _write_jsonl(records, out_path, overwrite):
                    summary_rows.append({
                        "dataset": dataset, "language": language,
                        "split": split, "n": len(records), "path": str(out_path),
                    })

    # Step 2: mixed files
    produced_splits = set()
    for dataset in DATASET_REGISTRY:
        for split in ALL_SPLITS:
            if _split_enabled(dataset, split):
                produced_splits.add(split)

    for split in ALL_SPLITS:
        if split not in produced_splits:
            continue
        if split == "train":
            lang_cap = max_per_language_mixed
        elif split == "val":
            lang_cap = DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL
        else:
            lang_cap = None  # no cap on test mixed

        mixed = _build_mixed(
            per_dir=per_dir,
            split=split,
            global_max_per_language=max_per_language,
            global_max_per_dataset=max_per_dataset,
            max_per_language_mixed=lang_cap,
            rng=rng,
            debug=debug,
        )
        if not mixed:
            continue
        mixed_path = mixed_dir / f"{split}_mixed.jsonl"
        if _write_jsonl(mixed, mixed_path, overwrite):
            summary_rows.append({
                "dataset": "mixed", "language": "all",
                "split": split, "n": len(mixed), "path": str(mixed_path),
            })

    # Step 3: summary CSV
    summary_path = output_dir / "build_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "language", "split", "n", "path"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary: {summary_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_language",       type=int, default=DEFAULT_MAX_PER_LANGUAGE)
    parser.add_argument("--max_per_dataset",        type=int, default=DEFAULT_MAX_PER_DATASET)
    parser.add_argument("--max_per_language_mixed", type=int, default=DEFAULT_MAX_PER_LANGUAGE_MIXED)
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config",    type=Path, default=None,
                        help="JSON config overriding DATASET_REGISTRY and global caps.")
    return parser.parse_args()


def main():
    args = parse_args()

    max_per_language = args.max_per_language
    max_per_dataset  = args.max_per_dataset
    max_mixed         = args.max_per_language_mixed if args.max_per_language_mixed > 0 else None
    output_dir        = DEFAULT_SFT_BASE_DIR

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        max_per_language = cfg.get("max_per_language", max_per_language)
        max_per_dataset  = cfg.get("max_per_dataset",  max_per_dataset)
        max_mixed        = cfg.get("max_per_language_mixed", max_mixed)
        if output_name := cfg.get("output_name"):
            output_dir = DEFAULT_SFT_BASE_DIR / output_name
        if "dataset_registry" in cfg:
            DATASET_REGISTRY.update(cfg["dataset_registry"])

    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"Debug:      {args.debug}  Overwrite: {args.overwrite}")
    print(f"Config:     {args.config or 'default'}")
    print(f"Output dir: {output_dir}")
    print(f"Caps:       max_per_language={max_per_language}, "
          f"max_per_dataset={max_per_dataset}, "
          f"max_per_language_mixed={max_mixed}\n")

    build(
        output_dir=output_dir,
        max_per_language=max_per_language,
        max_per_dataset=max_per_dataset,
        max_per_language_mixed=max_mixed,
        debug=args.debug,
        overwrite=args.overwrite,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()