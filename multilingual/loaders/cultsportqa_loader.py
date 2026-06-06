"""Load CultSportQA examples from processed JSONL files."""

from typing import Any, Dict, List, Optional, Tuple
import json
import re

import pandas as pd

from multilingual.config import REPO
from multilingual.utils import VAL_FRACTION, VAL_SEED


_HIRAGANA_KATAKANA = re.compile("[\u3040-\u30ff]")
PROCESSED_DIR = REPO / "multilingual" / "data" / "processed" / "cultsportqa"
SUPPORTED_LANGUAGES = {"chinese", "hindi"}


def _jsonl_path(language: str):
    return PROCESSED_DIR / f"cultsportqa_{language}_all.jsonl"


def _read_jsonl(path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_language(language: str) -> List[Dict[str, Any]]:
    path = _jsonl_path(language)
    if not path.exists():
        return []

    records = _read_jsonl(path)

    if language == "chinese":
        records = [
            record
            for record in records
            if not _HIRAGANA_KATAKANA.search(str(record.get("prompt", "")))
        ]

    if not records:
        return []

    df = pd.DataFrame({"record": records})
    df = df.sample(frac=1, random_state=VAL_SEED).reset_index(drop=True)
    return df["record"].tolist()


def _split(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n_val = max(1, int(len(records) * VAL_FRACTION))
    return records[n_val:], records[:n_val]


def _format_example(record: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    metadata = record.get("metadata", {})

    return {
        "id": f"cultsportqa_{language}_{idx}",
        "dataset": "cultsportqa",
        "language": language,
        "task_type": "mcq",
        "prompt": str(record["prompt"]),
        "answer": str(record["answer"]).strip().upper(),
        "metadata": {
            "country": str(metadata.get("country", "")),
            "sport": str(metadata.get("sport", "")),
            "group": str(metadata.get("group", "")),
            "group_description": str(metadata.get("group_description", "")),
            "row_idx": metadata.get("row_idx"),
            "source_idx": metadata.get("source_idx"),
            "source_id": record.get("source_id", record.get("id", "")),
        },
    }


def load_cultsportqa_examples(language: str, n: Optional[int] = None, split: str = "val") -> List[Dict[str, Any]]:
    """Load CultSportQA examples. split='val' or 'train'."""
    if language not in SUPPORTED_LANGUAGES:
        print(f"[CultSportQA] No data for {language}; skipping.")
        return []

    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    records = _load_language(language)
    if not records:
        print(f"[CultSportQA] Empty after filtering for {language}.")
        return []

    train_records, val_records = _split(records)
    data = val_records if split == "val" else train_records

    if n is not None:
        data = data[:n]

    return [_format_example(data[i], language, i) for i in range(len(data))]


def load_cultsportqa_splits(language: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (train_examples, val_examples)."""
    train = load_cultsportqa_examples(language, n=None, split="train")
    val = load_cultsportqa_examples(language, n=None, split="val")
    return train, val
