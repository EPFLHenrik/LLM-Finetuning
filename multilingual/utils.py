"""multilingual/utils.py"""
from pathlib import Path
from typing import Any, Optional, Tuple
import pandas as pd
from datasets import Dataset
from multilingual.config import VAL_FRACTION, VAL_SEED


def pick_split(ds) -> str:
    """Return the most informative available split name."""
    for split in ["test", "validation", "dev", "train"]:
        if split in ds:
            return split
    return list(ds.keys())[0]


def get_train_val_split(ds, split_name: str) -> Tuple[Dataset, Dataset]:
    """
    Return (train_pool, val_set) for a dataset split.

    If a dedicated train split exists alongside the test/val split we picked,
    use that as the train pool and the test split as val.

    Otherwise (most common case for INCLUDE and MGSM which only have 'test'),
    carve out VAL_FRACTION of the test split as val and use the rest as train.

    The split is deterministic via VAL_SEED so it never changes between runs.
    """
    data = ds[split_name]
    data = data.shuffle(seed=VAL_SEED)

    # If a separate training split exists, use it directly
    if "train" in ds and split_name != "train":
        train_pool = ds["train"].shuffle(seed=VAL_SEED)
        val_set = data          # the test/val split is purely for validation
        return train_pool, val_set

    # Otherwise split the single available split
    n_total = len(data)
    n_val = max(1, int(n_total * VAL_FRACTION))
    n_train = n_total - n_val

    val_set   = data.select(range(n_val))
    train_pool = data.select(range(n_val, n_val + n_train))
    return train_pool, val_set


def select_at_most(dataset: Dataset, n: Optional[int]) -> Dataset:
    """Return up to n examples from a (pre-shuffled) dataset."""
    if n is None:
        return dataset
    return dataset.select(range(min(n, len(dataset))))


def normalize_mc_answer(x: Any) -> Optional[str]:
    if x is None:
        return None
    return str(x).strip().upper()


def normalize_text_answer(x: Any) -> Optional[str]:
    if x is None:
        return None
    return str(x).strip()


def save_jsonl(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved: {out_path}")
    return out_path