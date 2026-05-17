"""multilingual/loaders/mera_loader.py"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from multilingual.config import REPO
from multilingual.utils import VAL_SEED

CLEAN_CSV = REPO / "multilingual" / "data" / "processed" / "mera" / "mera_clean_all.csv"
SUPPORTED_LANGUAGES = {"russian"}
SUPPORTED_SUBSETS = {
    "mamuramu",
    "mathlogicqa",
    "parus",
    "rcb",
    "ruopenbookqa",
    "ruworldtree",
    "rwsd",
    "use",
}


def _parse_metadata(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None or pd.isna(x):
        return {}
    return json.loads(str(x))


def _load_split(split: str) -> pd.DataFrame:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    df = pd.read_csv(CLEAN_CSV, encoding="utf-8")
    df = df[df["split"] == split].copy()
    df = df[df["subset"].isin(SUPPORTED_SUBSETS)].copy()
    if "language" in df.columns:
        df = df[df["language"] == "russian"].copy()

    return df.sample(frac=1, random_state=VAL_SEED).reset_index(drop=True)


def _builder_task_type(task_type: str) -> str:
    task_type = str(task_type).strip().lower()
    if task_type == "mcq":
        return "mcq"
    if task_type == "numeric":
        # USE answers are often strings like "1,4" or "8,1,9,7".
        # In the SFT builder, use math-style boxed answers for these.
        return "math"

    raise ValueError(f"Unexpected MERA task_type: {task_type!r}")


def _format_example(row: pd.Series, idx: int) -> Dict[str, Any]:
    subset = str(row["subset"]).strip().lower()
    raw_task_type = str(row["task_type"]).strip().lower()
    task_type = _builder_task_type(raw_task_type)
    prompt = str(row["prompt"]).strip()
    answer = str(row["answer"]).strip()
    if subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"Unsupported MERA subset: {subset!r}")
    if not prompt:
        raise ValueError("Empty MERA prompt.")
    if not answer:
        raise ValueError("Empty MERA answer.")
    metadata = _parse_metadata(row.get("metadata", "{}"))
    metadata.update({
        "subset": subset,
        "source_id": str(row.get("source_id", "")),
        "raw_task_type": raw_task_type,
    })
    return {
        "id": f"mera_{subset}_{idx}",
        "dataset": "mera",
        "language": "russian",
        "task_type": task_type,
        "prompt": prompt,
        "answer": answer,
        "metadata": metadata,
    }


def load_mera_examples(language: str, n: Optional[int] = None, split: str = "val") -> List[Dict[str, Any]]:
    """Load cleaned MERA examples. split='train' or 'val'."""
    if language not in SUPPORTED_LANGUAGES:
        print(f"[MERA] No data for {language}; skipping.")
        return []
    df = _load_split(split)
    if df.empty:
        print(f"[MERA] Empty split: {split}.")
        return []
    if n is not None:
        df = df.iloc[:n].copy()
    return [_format_example(df.iloc[i], i) for i in range(len(df))]


def load_mera_splits(language: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_mera_examples(language, n=None, split="train")
    val = load_mera_examples(language, n=None, split="val")
    return train, val