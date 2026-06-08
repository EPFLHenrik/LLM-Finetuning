"""Loader for the processed Quaderno Prefettizia MCQ dataset.

Returns normalized MCQ examples for the SFT builder.
The source file has no official train/val split, so this loader creates a
fixed deterministic train/val carve using VAL_SEED. No test split is produced;
benchmark_all.jsonl is treated as the test set by the SFT builder.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from multilinguality_model.config import VAL_SEED
from multilinguality_model.paths import get_repo_root, get_processed_data_dir
from multilinguality_model.data_pipeline.download_processed_data import ensure_processed_data_available

REPO_ROOT = get_repo_root()
DATA_PATH = get_processed_data_dir() / "quaderno_prefettizia" / "quaderno_prefettizia.jsonl"
DATASET_NAME = "quaderno_prefettizia"
LANGUAGE = "italian"
FALLBACK_PREFIX = "quaderno_prefettizia"
DEFAULT_VAL_FRACTION = 0.1


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    ensure_processed_data_available()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
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
    return f"{FALLBACK_PREFIX}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _normalize_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = str(record.get("prompt", "")).strip()
    answer = str(record.get("answer", record.get("mcq", {}).get("answer", ""))).strip().upper()

    if not prompt or not answer:
        return None
    if len(answer) != 1 or not answer.isalpha():
        return None

    source_id = str(record.get("source_id", "")).strip()
    if not source_id:
        source_id = _fallback_source_id(record, prompt, answer)

    metadata = dict(record.get("metadata") or {})
    for key in (
        "article_id",
        "article_title",
        "article_text",
        "valid",
        "generation",
        "verification",
        "mcq",
    ):
        if key in record and key not in metadata:
            metadata[key] = record[key]

    metadata.setdefault("original_dataset", record.get("dataset", DATASET_NAME))
    metadata.setdefault("original_id", record.get("id", source_id))
    metadata.setdefault("loader_source_file", str(DATA_PATH.relative_to(REPO_ROOT)))

    return {
        "id": str(record.get("id", source_id)),
        "source_id": source_id,
        "dataset": DATASET_NAME,
        "language": LANGUAGE,
        "task_type": str(record.get("task_type", "mcq")),
        "prompt": prompt,
        "answer": answer,
        "metadata": metadata,
    }


def _dedupe_by_source_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("source_id", "")).strip()
        if source_id in seen:
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

    n_val = max(1, int(n * val_fraction)) if val_fraction > 0 and n >= 10 else 0
    val_records = records[:n_val]
    train_records = records[n_val:]

    return train_records if split == "train" else val_records


def _cap(records: List[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    if n is None or n <= 0:
        return records
    return records[:n]


def load_quaderno_prefettizia_examples(
    language: str = LANGUAGE,
    n: Optional[int] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    if language != LANGUAGE:
        return []


    ensure_processed_data_available()
    raw_records = _load_jsonl(DATA_PATH)
    examples = [_normalize_record(record) for record in raw_records]
    examples = [example for example in examples if example is not None]
    examples = _dedupe_by_source_id(examples)
    examples = _split_records(examples, split=split)
    return _cap(examples, n)
