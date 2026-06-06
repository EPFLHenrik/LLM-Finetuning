"""Load MERA examples from processed JSONL files with deterministic recarving."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from multilingual.config import REPO
from multilingual.utils import VAL_FRACTION, VAL_SEED


CLEAN_JSONL = REPO / "multilingual" / "data" / "processed" / "mera" / "mera.jsonl"

SUPPORTED_LANGUAGES = {"russian"}
SUPPORTED_SUBSETS = {
    "mamuramu",
    "mathlogicqa",
    "parus",
    "ruopenbookqa",
    "ruworldtree",
}


def _read_jsonl(path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_all_records() -> List[Dict[str, Any]]:
    records = _read_jsonl(CLEAN_JSONL)
    records = [
        record
        for record in records
        if str(record.get("subset", "")).strip().lower() in SUPPORTED_SUBSETS
        and record.get("language") == "russian"
        and str(record.get("task_type", "")).strip().lower() == "mcq"
    ]
    return records


def _recarve_split(split: str) -> List[Dict[str, Any]]:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    records = _load_all_records()
    if not records:
        return []

    df = pd.DataFrame({"record": records})
    df = df.sample(frac=1, random_state=VAL_SEED).reset_index(drop=True)

    n_val = max(1, int(len(df) * VAL_FRACTION))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)

    selected = val_df if split == "val" else train_df
    return selected["record"].tolist()


def _builder_task_type(task_type: str) -> str:
    task_type = str(task_type).strip().lower()
    if task_type == "mcq":
        return "mcq"
    if task_type == "numeric":
        return "math"
    raise ValueError(f"Unexpected MERA task_type: {task_type!r}")


def _format_example(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    subset = str(record.get("subset", "")).strip().lower()
    raw_task_type = str(record.get("task_type", "")).strip().lower()
    task_type = _builder_task_type(raw_task_type)
    prompt = str(record.get("prompt", "")).strip()
    answer = str(record.get("answer", "")).strip()

    if subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"Unsupported MERA subset: {subset!r}")
    if not prompt:
        raise ValueError("Empty MERA prompt.")
    if not answer:
        raise ValueError("Empty MERA answer.")

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    metadata = dict(metadata)
    metadata.update(
        {
            "subset": subset,
            "source_id": str(record.get("source_id", "")),
            "raw_task_type": raw_task_type,
            "original_split": str(record.get("split", "")),
        }
    )

    return {
        "id": f"mera_{subset}_{idx}",
        "dataset": "mera",
        "language": "russian",
        "task_type": task_type,
        "prompt": prompt,
        "answer": answer,
        "metadata": metadata,
    }


def load_mera_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
) -> List[Dict[str, Any]]:
    """Load cleaned MERA examples using a deterministic train/val recarve."""
    if language not in SUPPORTED_LANGUAGES:
        print(f"[MERA] No data for {language}; skipping.")
        return []

    records = _recarve_split(split)
    if not records:
        print(f"[MERA] Empty split: {split}.")
        return []

    if n is not None:
        records = records[:n]

    return [_format_example(records[i], i) for i in range(len(records))]


def load_mera_splits(language: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (train_examples, val_examples)."""
    train = load_mera_examples(language, n=None, split="train")
    val = load_mera_examples(language, n=None, split="val")
    return train, val
