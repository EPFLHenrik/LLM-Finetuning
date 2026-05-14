"""multilingual/loaders/cultsportqa_loader.py"""
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from multilingual.config import REPO
from multilingual.utils import VAL_SEED, VAL_FRACTION
import re

_HIRAGANA_KATAKANA = re.compile(r'[\u3040-\u30ff]')
CLEAN_CSV = REPO / "multilingual" / "data" / "processed" / "cultsportqa" / "cultsportqa_clean_all.csv"
SUPPORTED_LANGUAGES = {"chinese", "hindi"}


def _load_language(language: str) -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV, encoding="utf-8")
    df = df[df["language"] == language].copy()
    if language == "chinese":
        df = df[~df["question"].astype(str).str.contains(_HIRAGANA_KATAKANA, na=False)].copy()
    return df.sample(frac=1, random_state=VAL_SEED).reset_index(drop=True)


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_val = max(1, int(len(df) * VAL_FRACTION))
    return df.iloc[n_val:].reset_index(drop=True), df.iloc[:n_val].reset_index(drop=True)


def _format_example(row: pd.Series, language: str, idx: int) -> Dict[str, Any]:
    return {
        "id":        f"cultsportqa_{language}_{idx}",
        "dataset":   "cultsportqa",
        "language":  language,
        "task_type": "mcq",
        "prompt":    str(row["prompt"]),
        "answer":    str(row["answer"]).strip().upper(),
        "metadata": {
            "country": str(row.get("country", "")),
            "sport": str(row.get("sport", "")),
            "group": str(row.get("group", "")),
            "group_description": str(row.get("group_description", "")),
            "row_idx": None if pd.isna(row.get("row_idx")) else int(row.get("row_idx")),
        },
    }


def load_cultsportqa_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
) -> List[Dict[str, Any]]:
    """Load CultSportQA examples. split='val' or 'train'."""
    if language not in SUPPORTED_LANGUAGES:
        print(f"[CultSportQA] No data for {language}; skipping.")
        return []
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    df = _load_language(language)
    if df.empty:
        print(f"[CultSportQA] Empty after filtering for {language}.")
        return []

    train_df, val_df = _split(df)
    data = val_df if split == "val" else train_df

    if n is not None:
        data = data.iloc[:n]
    return [_format_example(data.iloc[i], language, i) for i in range(len(data))]


def load_cultsportqa_splits(
    language: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_cultsportqa_examples(language, n=None, split="train")
    val   = load_cultsportqa_examples(language, n=None, split="val")
    return train, val