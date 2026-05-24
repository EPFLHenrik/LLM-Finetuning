"""multilingual/loaders/include_loader.py

Loads INCLUDE examples from local processed JSONL files (primary)
or HF (recarve mode only).

Processed files:
    multilingual/data/processed/include/include_{language}_all.jsonl

Split strategy (as_is mode, default):
    val:   first VAL_FRACTION of shuffled records (seed=VAL_SEED)
    train: remaining records after val carve-out
    test:  not produced in as_is mode

Recarve mode:
    Merges HF validation+test, recarves into val/test.
    Used when you need a specific val/test ratio from HF data.
    Train is not produced in recarve mode (train=0 in registry).
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from multilingual.config import LETTERS, REPO, TARGET_LANGUAGES
from multilingual.loaders.recarve_utils import recarve_splits
from multilingual.utils import VAL_SEED, select_at_most

try:
    from datasets import DatasetDict, load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

PROCESSED_DIR = REPO / "multilingual" / "data" / "processed" / "include"
VAL_FRACTION  = 0.30  # 30% val, 70% train from disk

_HF_SPLIT_MAP = [
    ("train",      "train"),
    ("validation", "val"),
    ("val",        "val"),
    ("test",       "test"),
    ("dev",        "val"),
]


# ---------------------------------------------------------------------------
# Disk loading
# ---------------------------------------------------------------------------

def _load_disk(language: str) -> List[Dict[str, Any]]:
    """Load all INCLUDE examples for a language from processed JSONL."""
    path = PROCESSED_DIR / f"include_{language}_all.jsonl"
    if not path.exists():
        print(f"[INCLUDE] Processed file not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _format_disk_record(r: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    """Reformat a processed disk record into the standard loader format."""
    return {
        "id":        f"include_{language}_{idx}",
        "source_id": str(r.get("source_id") or r.get("id") or f"include_{language}_{idx}"),
        "dataset":   "include",
        "language":  language,
        "task_type": "mcq",
        "prompt":    str(r["prompt"]),
        "answer":    str(r["answer"]).strip().upper(),
        "metadata":  r.get("metadata", {}),
    }


def _disk_train_val(
    language: str,
    n: Optional[int],
    split: str,
) -> List[Dict[str, Any]]:
    """Load train or val from disk with a fixed seed split."""
    records = _load_disk(language)
    if not records:
        return []

    rng = random.Random(VAL_SEED)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * VAL_FRACTION))
    val_records   = records[:n_val]
    train_records = records[n_val:]

    pool = val_records if split == "val" else train_records
    if n is not None:
        pool = pool[:n]

    return [_format_disk_record(r, language, i) for i, r in enumerate(pool)]


# ---------------------------------------------------------------------------
# HF loading (recarve mode only)
# ---------------------------------------------------------------------------

def _get_hf_splits(ds: "DatasetDict") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for hf_name, internal in _HF_SPLIT_MAP:
        if hf_name in ds and internal not in result:
            result[internal] = ds[hf_name]
    return result


def _build_prompt(ex: Dict[str, Any]) -> str:
    question = str(ex["question"])
    choices  = [str(ex[f"option_{l.lower()}"]) for l in LETTERS[:4]]
    options  = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{options}"


def _format_hf_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    return {
        "id":        f"include_{language}_{idx}",
        "dataset":   "include",
        "language":  language,
        "task_type": "mcq",
        "prompt":    _build_prompt(ex),
        "answer":    LETTERS[int(ex["answer"])],
        "metadata": {
            "language":         ex.get("language"),
            "country":          ex.get("country"),
            "domain":           ex.get("domain"),
            "subject":          ex.get("subject"),
            "regional_feature": ex.get("regional_feature"),
            "level":            ex.get("level"),
        },
    }


def _load_hf(language: str) -> Optional[Dict[str, Any]]:
    if not _HF_AVAILABLE:
        return None
    include_config = TARGET_LANGUAGES[language]["include_config"]
    if include_config is None:
        print(f"[INCLUDE] No config for {language}; skipping.")
        return None
    ds = load_dataset("CohereLabs/include-base-44", include_config)
    return _get_hf_splits(ds)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_include_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
    mode: str = "as_is",
) -> List[Dict[str, Any]]:
    """
    Load INCLUDE examples for a single split.

    as_is mode (default):
        Loads from processed disk files.
        val   → first VAL_FRACTION of shuffled records
        train → remaining records after val carve-out
        test  → not produced (returns [])

    recarve mode:
        Merges HF validation+test, recarves into val/test.
        train → not produced in recarve mode (use train=0 in registry)
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

    if mode == "recarve":
        if split == "train":
            return []
        n_val  = n if split == "val"  else None
        n_test = n if split == "test" else None
        val_ex, test_ex = load_include_recarved(language, n_val=n_val, n_test=n_test)
        return val_ex if split == "val" else test_ex

    # as_is: load from disk
    if split == "test":
        return []  # test not produced from disk in as_is mode

    return _disk_train_val(language, n, split)


def load_include_recarved(
    language: str,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge HF validation+test and recarve into (val_examples, test_examples)."""
    splits = _load_hf(language)
    if splits is None:
        return [], []

    to_merge = [splits[k] for k in ("val", "test") if k in splits]
    val_data, test_data = recarve_splits(to_merge, n_val, n_test, "INCLUDE", language)

    if val_data is None or test_data is None:
        return [], []

    val_ex  = [_format_hf_example(val_data[i],  language, i) for i in range(len(val_data))]
    test_ex = [_format_hf_example(test_data[i], language, i) for i in range(len(test_data))]
    return val_ex, test_ex