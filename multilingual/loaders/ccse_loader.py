"""Loader for the processed Instituto Cervantes CCSE dataset.

Uses:
    multilingual/data/processed/ccse/ccse_parsed.jsonl

The file contains 299 parsed Spanish MCQ examples from the CCSE 2026 manual.
The corrupted item ccse_4036 is already dropped by the parser. Since the dataset
is small and highly regional, the loader uses all examples for train by default
and produces no local val/test split unless split='all' is requested.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from multilingual.config import REPO, VAL_SEED

DATA_PATH = REPO / "multilingual/data/processed/ccse/ccse.jsonl"

DATASET_NAME = "ccse"
LANGUAGE = "spanish"
FALLBACK_PREFIX = "ccse"
DEFAULT_VAL_FRACTION = 0.0
VALID_ANSWER_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DROP_SOURCE_IDS = {"ccse_4036"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e

    return records


def _fallback_source_id(record: Dict[str, Any], prompt: str, answer: str) -> str:
    raw = "\n".join([
        DATASET_NAME,
        LANGUAGE,
        str(record.get("id", "")),
        prompt,
        answer,
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{FALLBACK_PREFIX}_{digest}"


def _normalize_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = str(record.get("prompt", "")).strip()
    answer = str(record.get("answer", "")).strip().upper()

    if not prompt or not answer:
        return None

    if answer not in VALID_ANSWER_LETTERS:
        return None

    if f"{answer}) " not in prompt:
        return None

    language = str(record.get("language", LANGUAGE)).strip()
    if language != LANGUAGE:
        return None

    dataset = str(record.get("dataset", DATASET_NAME)).strip()
    if dataset != DATASET_NAME:
        return None

    task_type = str(record.get("task_type", "mcq")).strip()
    if task_type != "mcq":
        return None

    source_id = str(record.get("source_id", "")).strip()
    if not source_id:
        source_id = _fallback_source_id(record, prompt, answer)

    if source_id in DROP_SOURCE_IDS:
        return None

    record_id = str(record.get("id", source_id)).strip() or source_id

    metadata = dict(record.get("metadata") or {})
    metadata.setdefault("original_dataset", record.get("dataset", DATASET_NAME))
    metadata.setdefault("original_id", record.get("id", source_id))
    metadata.setdefault("source_split", record.get("split", "parsed"))
    metadata.setdefault("loader_source_file", str(DATA_PATH.relative_to(REPO)))

    return {
        "id": record_id,
        "source_id": source_id,
        "dataset": DATASET_NAME,
        "language": LANGUAGE,
        "task_type": "mcq",
        "prompt": prompt,
        "answer": answer,
        "metadata": metadata,
    }


def _dedupe_by_source_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for record in records:
        source_id = str(record.get("source_id", "")).strip()
        if not source_id or source_id in seen:
            continue

        seen.add(source_id)
        out.append(record)

    return out


def _split_records(
    records: List[Dict[str, Any]],
    split: str,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> List[Dict[str, Any]]:
    if split not in {"train", "val", "all", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    records = list(records)

    rng = random.Random(VAL_SEED)
    rng.shuffle(records)

    if split == "all":
        return records

    if split == "test":
        return []

    n = len(records)
    if n == 0:
        return []

    if val_fraction <= 0:
        return records if split == "train" else []

    n_val = max(1, int(n * val_fraction)) if n >= 10 else 0
    val_records = records[:n_val]
    train_records = records[n_val:]

    return train_records if split == "train" else val_records


def _cap(records: List[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    if n is None or n <= 0:
        return records

    return records[:n]


def load_ccse_examples(
    language: str = LANGUAGE,
    n: Optional[int] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    if language != LANGUAGE:
        return []

    raw_records = _load_jsonl(DATA_PATH)

    examples = [_normalize_record(record) for record in raw_records]
    examples = [example for example in examples if example is not None]
    examples = _dedupe_by_source_id(examples)
    examples = _split_records(examples, split=split)

    return _cap(examples, n)


def load_ccse_splits(
    language: str = LANGUAGE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = load_ccse_examples(language=language, n=None, split="train")
    val = load_ccse_examples(language=language, n=None, split="val")
    return train, val