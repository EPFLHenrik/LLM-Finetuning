from pathlib import Path
from typing import Any, Optional

import pandas as pd


def pick_split(ds) -> str:
    for split in ["test", "validation", "dev", "train"]:
        if split in ds:
            return split
    return list(ds.keys())[0]


def normalize_mc_answer(x: Any) -> Optional[str]:
    if x is None:
        return None
    return str(x).strip().upper()


def normalize_text_answer(x: Any) -> Optional[str]:
    if x is None:
        return None
    return str(x).strip()


def select_at_most(dataset, n=None, seed: int = 42):
    dataset = dataset.shuffle(seed=seed)
    if n is None:
        return dataset
    n_eff = min(n, len(dataset))
    return dataset.select(range(n_eff))


def save_jsonl(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved: {out_path}")
    return out_path