"""multilingual/loaders/head_qa_loader.py"""

from typing import Any, Dict, List, Optional, Tuple
import json
import pandas as pd
from multilingual.config import REPO
from multilingual.utils import VAL_FRACTION, VAL_SEED

CLEAN_CSV = REPO / "multilingual" / "data" / "processed" / "head_qa" / "head_qa_clean_all.csv"
SUPPORTED_LANGUAGES = {"spanish", "italian"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_language(language: str) -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV, encoding="utf-8")
    df = df[df["language"] == language].copy()
    return df.sample(frac=1, random_state=VAL_SEED).reset_index(drop=True)


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_val = max(1, int(len(df) * VAL_FRACTION))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


def _parse_json_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    return json.loads(str(x))


def _format_example(row: pd.Series, language: str, idx: int) -> Dict[str, Any]:
    question = str(row["qtext"]).strip()
    options = [str(x).strip() for x in _parse_json_list(row["atexts"])]
    answer = str(row["ra_letter"]).strip().upper()
    if not question:
        raise ValueError("Empty HEAD-QA question.")
    if not (2 <= len(options) <= len(LETTERS)):
        raise ValueError(f"Unexpected number of HEAD-QA options: {len(options)}")
    if answer not in LETTERS[:len(options)]:
        raise ValueError(f"Answer {answer!r} incompatible with {len(options)} options.")

    options_text = "\n".join(
        f"{LETTERS[i]}) {option}"
        for i, option in enumerate(options)
    )

    prompt = f"{question}\n\n{options_text}"
    return {
        "id": f"head_qa_{language}_{idx}",
        "dataset": "head_qa",
        "language": language,
        "task_type": "mcq",
        "prompt": prompt,
        "answer": answer,
        "metadata": {
            "config": str(row.get("config", "")),
            "qid": int(row["qid"]) if not pd.isna(row.get("qid")) else None,
            "name": str(row.get("name", "")),
            "year": int(row["year"]) if not pd.isna(row.get("year")) else None,
            "category": str(row.get("category", "")),
            "n_answers": int(row["n_answers"]) if not pd.isna(row.get("n_answers")) else len(options),
            "translated": language == "italian",
        },
    }


def load_head_qa_examples(language: str, n: Optional[int] = None, split: str = "val") -> List[Dict[str, Any]]:
    """Load HEAD-QA examples. split='train' or 'val'."""
    if language not in SUPPORTED_LANGUAGES:
        print(f"[HEAD-QA] No data for {language}; skipping.")
        return []
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    df = _load_language(language)
    if df.empty:
        print(f"[HEAD-QA] Empty after filtering for {language}.")
        return []
    train_df, val_df = _split(df)
    data = val_df if split == "val" else train_df
    if n is not None:
        data = data.iloc[:n]
    return [_format_example(data.iloc[i], language, i) for i in range(len(data))]


def load_head_qa_splits(language: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = load_head_qa_examples(language, n=None, split="train")
    val = load_head_qa_examples(language, n=None, split="val")
    return train, val