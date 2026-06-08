"""multilingual/loaders/ceval_loader.py

Loads C-Eval examples from the preprocessed local JSONL file.
No HuggingFace calls at build time.

Preprocessing was done once via notebook, producing:
    data/multilinguality/sft/preprocessed/ceval_chinese_all.jsonl

Fields available for filtering:
    category:    STEM | Humanities | Social Science | Law & Professional | Other
    regionality: regional | regional_implicit | region_agnostic

Pipeline order: read all → filter → stratified split → cap.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multilinguality_model.paths import get_processed_data_dir
from multilinguality_model.data_pipeline.download_processed_data import ensure_processed_data_available

PREPROCESSED_PATH = get_processed_data_dir() / "ceval" / "ceval_chinese_all.jsonl"

# Default split ratios (train / val)
DEFAULT_VAL_FRACTION = 0.20

# Default regionality weights for sampling (1.0 = keep all, 0.0 = skip)
DEFAULT_REGIONALITY_WEIGHTS = {
    "regional":          1.0,
    "regional_implicit": 1.0,
    "region_agnostic":   0.3,
}

# Default category weights
DEFAULT_CATEGORY_WEIGHTS = {
    "Law & Professional": 1.0,
    "Social Science":     1.0,
    "Humanities":         1.0,
    "STEM":               0.3,
    "Other":              0.5,
}

VAL_SEED = 42


def _load_all() -> List[Dict[str, Any]]:
    """Load the full preprocessed CEval JSONL from disk."""
    if not PREPROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"CEval preprocessed file not found: {PREPROCESSED_PATH}\n"
            f"Run the preprocessing notebook first."
        )
    import json
    records = []
    with open(PREPROCESSED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _apply_weights(
    records: List[Dict[str, Any]],
    regionality_weights: Dict[str, float],
    category_weights: Dict[str, float],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Probabilistically filter records by regionality and category weights."""
    kept = []
    for r in records:
        reg = r["metadata"].get("regionality", "region_agnostic")
        cat = r["metadata"].get("category", "Other")
        w   = regionality_weights.get(reg, 1.0) * category_weights.get(cat, 1.0)
        if w >= 1.0 or rng.random() < w:
            kept.append(r)
    return kept


def _stratified_split(
    records: List[Dict[str, Any]],
    val_fraction: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified train/val split by subject, minimum 1 val example per subject."""
    by_subject: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        subject = r["metadata"].get("subject", "unknown")
        by_subject[subject].append(r)

    train, val = [], []
    for pairs in by_subject.values():
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def _reformat(r: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    """Return record with correct split label and stable source_id."""
    source_id = str(
        r.get("source_id")
        or r.get("id")
        or r.get("metadata", {}).get("source_id")
        or f"ceval_chinese_{split}_{idx}"
    ).strip()

    metadata = dict(r.get("metadata") or {})
    metadata.setdefault("source_id", source_id)

    return {
        **r,
        "id": f"ceval_chinese_{split}_{idx}",
        "source_id": source_id,
        "split": split,
        "metadata": metadata,
    }


def _hard_filter(
    records: List[Dict[str, Any]],
    regionality: Optional[List[str]],
    category: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Hard filter by regionality and/or category lists (exact match)."""
    if regionality:
        reg_set = set(regionality)
        records = [r for r in records
                   if r["metadata"].get("regionality") in reg_set]
    if category:
        cat_set = set(category)
        records = [r for r in records
                   if r["metadata"].get("category") in cat_set]
    return records


def load_ceval_examples(
    language: str = "chinese",
    n: Optional[int] = None,
    split: str = "val",
    val_fraction: float = DEFAULT_VAL_FRACTION,
    regionality: Optional[List[str]] = None,
    category: Optional[List[str]] = None,
    regionality_weights: Optional[Dict[str, float]] = None,
    category_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Load C-Eval examples from disk.

    Args:
        n:                    Max examples to return (None = all).
        split:                'train' or 'val'.
        val_fraction:         Fraction of each subject reserved for val.
        regionality:          Hard filter — keep only these regionality values,
                              e.g. ["regional", "regional_implicit"].
                              Applied before weight-based sampling.
        category:             Hard filter — keep only these category values,
                              e.g. ["Humanities", "Law & Professional"].
        regionality_weights:  Probabilistic sampling weights per regionality (0–1).
                              Applied after hard filter.
        category_weights:     Probabilistic sampling weights per category (0–1).

    Returns:
        List of SFT-format example dicts.
    """
    if language != "chinese":
        return []

    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    rng   = random.Random(VAL_SEED)
    reg_w = regionality_weights or DEFAULT_REGIONALITY_WEIGHTS
    cat_w = category_weights    or DEFAULT_CATEGORY_WEIGHTS

    records = _load_all()
    records = _hard_filter(records, regionality, category)
    records = _apply_weights(records, reg_w, cat_w, rng)

    train, val = _stratified_split(records, val_fraction, rng)
    pool = val if split == "val" else train

    rng.shuffle(pool)
    if n is not None:
        pool = pool[:n]

    return [_reformat(r, split, i) for i, r in enumerate(pool)]


def load_ceval_splits(
    val_fraction: float = DEFAULT_VAL_FRACTION,
    regionality: Optional[List[str]] = None,
    category: Optional[List[str]] = None,
    regionality_weights: Optional[Dict[str, float]] = None,
    category_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples) in one pass."""
    rng   = random.Random(VAL_SEED)
    reg_w = regionality_weights or DEFAULT_REGIONALITY_WEIGHTS
    cat_w = category_weights    or DEFAULT_CATEGORY_WEIGHTS

    records = _load_all()
    records = _hard_filter(records, regionality, category)
    records = _apply_weights(records, reg_w, cat_w, rng)
    train, val = _stratified_split(records, val_fraction, rng)

    rng.shuffle(train)
    rng.shuffle(val)

    train = [_reformat(r, "train", i) for i, r in enumerate(train)]
    val   = [_reformat(r, "val",   i) for i, r in enumerate(val)]
    return train, val
