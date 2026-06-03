"""multilingual/loaders/exams_loader.py

Loader for processed EXAMS train JSONL files.

The *_eval.jsonl files are reserved for benchmark construction and are not
loaded here.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from multilingual.config import REPO
from multilingual.utils import VAL_FRACTION, VAL_SEED

EXAMS_DIR = REPO / "multilingual" / "data" / "processed" / "exams"
SUPPORTED_LANGUAGES = {"italian", "spanish"}
VALID_ANSWER_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _jsonl_path(language: str) -> Path:
    return EXAMS_DIR / f"exams_{language}_train.jsonl"


def _fallback_source_id(record: Dict[str, Any], language: str) -> str:
    raw = "\n".join([
        "exams",
        language,
        str(record.get("id", "")),
        str(record.get("prompt", "")).strip(),
        str(record.get("answer", "")).strip().upper(),
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"exams_{language}_{digest}"


def _validate_record(record: Dict[str, Any], language: str) -> bool:
    if record.get("dataset") != "exams":
        return False
    if record.get("language") != language:
        return False
    if record.get("task_type") != "mcq":
        return False

    prompt = str(record.get("prompt", "")).strip()
    answer = str(record.get("answer", "")).strip().upper()

    if not prompt or not answer:
        return False
    if answer not in VALID_ANSWER_LETTERS:
        return False
    if "A)" not in prompt or "B)" not in prompt:
        return False

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return False

    return True


def _normalize_record(record: Dict[str, Any], language: str) -> Optional[Dict[str, Any]]:
    if not _validate_record(record, language):
        return None

    prompt = str(record["prompt"]).strip()
    answer = str(record["answer"]).strip().upper()

    source_id = str(record.get("source_id", "")).strip()
    metadata = dict(record.get("metadata") or {})

    if not source_id:
        source_id = str(metadata.get("source_id", "")).strip()

    if not source_id:
        source_id = _fallback_source_id(record, language)

    record_id = str(record.get("id", source_id)).strip() or source_id
    metadata.setdefault("original_dataset", record.get("dataset", "exams"))
    metadata.setdefault("original_id", record.get("id", source_id))
    metadata.setdefault("source_split", record.get("split", metadata.get("source_split", "train")))

    return {
        "id": record_id,
        "source_id": source_id,
        "dataset": "exams",
        "language": language,
        "task_type": "mcq",
        "prompt": prompt,
        "answer": answer,
        "metadata": metadata,
    }


def _load_train_records(language: str) -> List[Dict[str, Any]]:
    if language not in SUPPORTED_LANGUAGES:
        print(f"[EXAMS] No data for {language}; skipping.")
        return []

    path = _jsonl_path(language)
    if not path.exists():
        print(f"[EXAMS] Missing processed train JSONL for {language}: {path}")
        return []

    examples: List[Dict[str, Any]] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            example = _normalize_record(record, language)
            if example is None:
                skipped += 1
                continue

            examples.append(example)

    if skipped:
        print(f"[EXAMS] Skipped {skipped} invalid rows for {language}.")
    if not examples:
        print(f"[EXAMS] Empty processed train JSONL for {language}: {path}")

    return _dedupe_by_source_id(examples)


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


def _split_records(records: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    if split not in {"train", "val", "all", "test"}:
        raise ValueError(f"Unsupported EXAMS split: {split!r}")

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

    n_val = max(1, int(n * VAL_FRACTION)) if n >= 10 else 0
    val_records = records[:n_val]
    train_records = records[n_val:]

    return train_records if split == "train" else val_records


def _cap(records: List[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    if n is None or n <= 0:
        return records
    return records[:n]


def load_exams_examples(language: str, n: Optional[int] = None, split: str = "val", mode: str = "as_is") -> List[Dict[str, Any]]:
    if mode not in {"as_is", "recarve"}:
        raise ValueError(f"Unsupported EXAMS mode: {mode!r}")
    if mode == "recarve":
        print("[EXAMS] recarve mode is no longer used; loading train JSONL with local train/val split.")

    records = _load_train_records(language)
    records = _split_records(records, split)
    return _cap(records, n)


def load_exams_recarved(language: str, n_val: Optional[int] = None, n_test: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = _load_train_records(language)
    rng = random.Random(VAL_SEED)
    rng.shuffle(records)

    n_val_final = n_val if n_val is not None else max(1, int(len(records) * VAL_FRACTION))
    val_records = records[:n_val_final]

    if n_test is None or n_test <= 0:
        test_records: List[Dict[str, Any]] = []
    else:
        test_records = records[n_val_final:n_val_final + n_test]

    return val_records, test_records


def load_exams_splits(language: str, mode: str = "as_is") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = load_exams_examples(language, n=None, split="train", mode=mode)
    val = load_exams_examples(language, n=None, split="val", mode=mode)
    return train, val