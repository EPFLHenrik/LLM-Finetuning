"""multilingual/loaders/exams_loader.py"""

from __future__ import annotations
import ast
import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from multilingual.config import REPO
from multilingual.utils import VAL_SEED

CLEAN_CSV = REPO / "multilingual" / "data" / "processed" / "exams" / "exams_clean_all.csv"
SUPPORTED_LANGUAGES = {"italian", "spanish"}
LANGUAGE_MAP = {
    "Italian": "italian",
    "Spanish": "spanish",
    "italian": "italian",
    "spanish": "spanish",
}

def _parse_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v).strip() for v in x]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = str(x).strip()
    # handle double-encoded JSON: '"[\"A\", \"B\"]"' -> '["A", "B"]' -> [...]
    try:
        parsed = json.loads(s)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
    except (json.JSONDecodeError, ValueError):
        parsed = ast.literal_eval(s)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)}")
    return [str(v).strip() for v in parsed]

def _safe_int(x: Any) -> Optional[int]:
    if x is None or pd.isna(x):
        return None
    return int(x)

def _normalize_language(x: Any) -> str:
    lang = str(x).strip()
    return LANGUAGE_MAP.get(lang, lang.lower())


def _load_split(language: str, split: str) -> pd.DataFrame:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    df = pd.read_csv(CLEAN_CSV, encoding="utf-8")
    df["language"] = df["language"].apply(_normalize_language)
    df = df[df["language"] == language].copy()
    if "split" not in df.columns:
        raise ValueError("EXAMS clean CSV must contain a split column.")
    split_values = {"train": "train", "validation": "val", "val": "val"}
    df["split"] = df["split"].map(lambda x: split_values.get(str(x).strip(), str(x).strip()))
    df = df[df["split"] == split].copy()
    return df.sample(frac=1, random_state=VAL_SEED).reset_index(drop=True)


def _format_example(row: pd.Series, language: str, idx: int) -> Dict[str, Any]:
    stem = str(row["stem"]).strip()
    labels = _parse_list(row["labels"])
    texts = _parse_list(row["texts"])
    answer = str(row["answerKey"]).strip().upper()

    if not stem:
        raise ValueError("Empty EXAMS question.")
    if len(labels) != len(texts):
        raise ValueError(f"labels/texts length mismatch: {len(labels)} vs {len(texts)}")
    if not (2 <= len(labels) <= 5):
        raise ValueError(f"Unexpected number of EXAMS choices: {len(labels)}")
    if answer not in labels:
        raise ValueError(f"Answer {answer!r} not in labels {labels!r}")
    if any(not str(t).strip() for t in texts):
        raise ValueError("Empty EXAMS option text.")

    options_text = "\n".join(f"{label}) {text}" for label, text in zip(labels, texts))
    return {
        "id": f"exams_{language}_{idx}",
        "dataset": "exams",
        "language": language,
        "task_type": "mcq",
        "prompt": f"{stem}\n\n{options_text}",
        "answer": answer,
        "metadata": {
            "source_id": str(row.get("id", "")),
            "subject": str(row.get("subject", "")),
            "grade": _safe_int(row.get("grade")),
            "n_choices": len(labels),
            "original_split": str(row.get("split", "")),
        },
    }


def load_exams_examples(language: str, n: Optional[int] = None, split: str = "val") -> List[Dict[str, Any]]:
    """Load cleaned EXAMS examples. split='train' or 'val'."""
    if language not in SUPPORTED_LANGUAGES:
        print(f"[EXAMS] No data for {language}; skipping.")
        return []
    df = _load_split(language, split)
    if df.empty:
        print(f"[EXAMS] Empty after filtering for language={language}, split={split}.")
        return []

    if n is not None:
        df = df.iloc[:n].copy()
    return [_format_example(df.iloc[i], language, i) for i in range(len(df))]


def load_exams_splits(language: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (train_examples, val_examples)."""
    train = load_exams_examples(language, n=None, split="train")
    val = load_exams_examples(language, n=None, split="val")
    return train, val