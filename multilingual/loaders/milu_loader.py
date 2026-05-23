"""multilingual/loaders/milu_loader.py

Loads MILU Hindi examples from preprocessed local JSONL files.
No HuggingFace calls at build time.

Preprocessing saved:
    multilingual/data/sft/preprocessed/milu_hindi_train.jsonl  (source: HF "test")
    multilingual/data/sft/preprocessed/milu_hindi_val.jsonl    (source: HF "validation")
    multilingual/data/sft/preprocessed/milu_hindi_all.jsonl    (both combined)

Two loading modes:
  preserved (default): respect original HF splits
    split="train" → milu_hindi_train.jsonl
    split="val"   → milu_hindi_val.jsonl
    split="all"   → milu_hindi_all.jsonl

  recarve: merge train+val, then recarve deterministically by custom ratios
    useful when custom split sizes are needed

Filtering:
  regionality: list of values to keep, e.g. ["regional", "regional_implicit"]
  domain:      list of domain strings to keep (None = keep all)

Stratification (recarve mode only):
  stratify_by: metadata key or list of keys used for stratified splitting
               e.g. "subject", "domain", "regionality", or ["subject", "domain"]
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

REPO = Path(__file__).resolve().parents[2]
PREPROC_DIR = REPO / "multilingual" / "data" / "processed" / "milu"

TRAIN_FILE = PREPROC_DIR / "milu_hindi_train.jsonl"
VAL_FILE   = PREPROC_DIR / "milu_hindi_val.jsonl"
ALL_FILE   = PREPROC_DIR / "milu_hindi_all.jsonl"

VAL_SEED = 42

# Recommended distillation rates (not used by loader, kept as reference)
DISTILLATION_WEIGHTS = {
    "regional":          1.00,
    "regional_implicit": 0.40,
    "region_agnostic":   0.05,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"MILU preprocessed file not found: {path}\n"
            f"Run the MILU preprocessing notebook first."
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _reformat(r: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    """Return record with updated split label and id."""
    return {**r, "id": f"milu_hindi_{split}_{idx}", "split": split}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter(
    records: List[Dict[str, Any]],
    regionality: Optional[List[str]],
    domain: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if regionality:
        reg_set = set(regionality)
        records = [r for r in records
                   if r["metadata"].get("regionality") in reg_set]
    if domain:
        dom_set = set(domain)
        records = [r for r in records
                   if r["metadata"].get("domain") in dom_set]
    return records


# ---------------------------------------------------------------------------
# Stratified recarve
# ---------------------------------------------------------------------------

def _stratum_key(r: Dict[str, Any], stratify_by: Union[str, List[str]]) -> str:
    """Derive a group key from one or more metadata fields."""
    if isinstance(stratify_by, str):
        stratify_by = [stratify_by]
    parts = [str(r["metadata"].get(k, "unknown")) for k in stratify_by]
    return "|".join(parts)


def _recarve_split(
    records: List[Dict[str, Any]],
    val_fraction: float,
    test_fraction: float,
    stratify_by: Optional[Union[str, List[str]]],
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Recarve records into (train, val, test) deterministically.
    test_fraction=0.0 → no test split.
    Handles small strata gracefully (guarantees at least 1 val if stratum >= 2).
    """
    rng = random.Random(seed)

    if stratify_by:
        by_stratum: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            by_stratum[_stratum_key(r, stratify_by)].append(r)
        groups = list(by_stratum.values())
    else:
        groups = [records]

    train, val, test = [], [], []
    for group in groups:
        g = group[:]
        rng.shuffle(g)
        n = len(g)
        if n == 1:
            train.extend(g)
            continue
        n_val  = max(1, round(n * val_fraction))
        n_test = max(0, round(n * test_fraction)) if test_fraction > 0 else 0
        n_val  = min(n_val,  n - 1)
        n_test = min(n_test, n - n_val - 1) if n_val + n_test < n else 0
        val.extend(g[:n_val])
        test.extend(g[n_val:n_val + n_test])
        train.extend(g[n_val + n_test:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_milu_examples(
    n: Optional[int] = None,
    split: str = "val",
    # Filtering
    regionality: Optional[List[str]] = None,
    domain: Optional[List[str]] = None,
    # Recarve mode
    recarve: bool = False,
    val_fraction: float = 0.15,
    test_fraction: float = 0.0,
    stratify_by: Optional[Union[str, List[str]]] = None,
    seed: int = VAL_SEED,
    # HF token kept for API compatibility (unused)
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load MILU Hindi examples from disk.

    Args:
        n:             Max examples (None = all).
        split:         'train', 'val', or 'all'.
        regionality:   Filter by regionality values, e.g. ["regional"].
        domain:        Filter by domain strings.
        recarve:       If True, merge all data and recarve by val/test fractions.
        val_fraction:  Val fraction for recarve mode.
        test_fraction: Test fraction for recarve mode (0.0 = no test split).
        stratify_by:   Metadata key(s) for stratified recarve.
        seed:          Random seed for recarve.
        hf_token:      Ignored (kept for backwards compatibility).
    """
    if split not in {"train", "val", "all"}:
        raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")

    if recarve:
        records = _read_jsonl(ALL_FILE)
        records = _filter(records, regionality, domain)
        train, val, _ = _recarve_split(
            records, val_fraction, test_fraction, stratify_by, seed
        )
        pool = {"train": train, "val": val, "all": records}[split]
    else:
        path = {"train": TRAIN_FILE, "val": VAL_FILE, "all": ALL_FILE}[split]
        pool = _read_jsonl(path)
        pool = _filter(pool, regionality, domain)
        rng  = random.Random(seed)
        rng.shuffle(pool)

    if n is not None:
        pool = pool[:n]

    return [_reformat(r, split, i) for i, r in enumerate(pool)]


def load_milu_splits(
    regionality: Optional[List[str]] = None,
    domain: Optional[List[str]] = None,
    recarve: bool = False,
    val_fraction: float = 0.15,
    stratify_by: Optional[Union[str, List[str]]] = None,
    seed: int = VAL_SEED,
    hf_token: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_milu_examples(
        split="train", regionality=regionality, domain=domain,
        recarve=recarve, val_fraction=val_fraction,
        stratify_by=stratify_by, seed=seed,
    )
    val = load_milu_examples(
        split="val", regionality=regionality, domain=domain,
        recarve=recarve, val_fraction=val_fraction,
        stratify_by=stratify_by, seed=seed,
    )
    return train, val