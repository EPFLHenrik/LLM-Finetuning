"""multilingual/loaders/exams_loader.py"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset
from multilingual.config import REPO
from multilingual.loaders.recarve_utils import recarve_splits
from multilingual.utils import VAL_SEED, select_at_most

EXAMS_DIR = REPO / "multilingual" / "data" / "processed" / "exams"
SUPPORTED_LANGUAGES = {"italian", "spanish"}
_SOURCE_SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}


def _jsonl_path(language: str) -> Path:
    return EXAMS_DIR / f"exams_{language}_all.jsonl"


def _normalize_source_split(x: Any) -> str:
    return _SOURCE_SPLIT_MAP.get(str(x).strip(), str(x).strip())


def _validate_example(ex: Dict[str, Any]) -> bool:
    required = ["id", "dataset", "language", "task_type", "prompt", "answer", "metadata"]
    for key in required:
        if key not in ex:
            return False
    if ex["dataset"] != "exams":
        return False
    if ex["task_type"] != "mcq":
        return False
    if ex["answer"] not in {"A", "B", "C", "D", "E"}:
        return False
    if not str(ex["prompt"]).strip():
        return False
    if "A)" not in ex["prompt"] or "B)" not in ex["prompt"]:
        return False
    if not isinstance(ex["metadata"], dict):
        return False
    return True


def _load_all_examples(language: str) -> List[Dict[str, Any]]:
    if language not in SUPPORTED_LANGUAGES:
        print(f"[EXAMS] No data for {language}; skipping.")
        return []

    path = _jsonl_path(language)
    if not path.exists():
        print(f"[EXAMS] Missing processed JSONL file for {language}: {path}")
        return []

    examples, skipped = [], 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not _validate_example(ex):
                skipped += 1
                continue
            ex["language"] = language
            examples.append(ex)

    if skipped:
        print(f"[EXAMS] Skipped {skipped} invalid JSONL rows for {language}.")
    if not examples:
        print(f"[EXAMS] Empty processed JSONL for {language}: {path}")
        return []

    print(f"[EXAMS] Loaded {len(examples)} processed examples for {language}.")
    return examples


def _examples_to_dataset(examples: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(examples)


def _dataset_to_examples(data) -> List[Dict[str, Any]]:
    return [dict(row) for row in data]


def _filter_as_is_split(
    examples: List[Dict[str, Any]],
    split: str,
) -> List[Dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

    selected = []
    for ex in examples:
        meta = ex.get("metadata", {})
        source_split = _normalize_source_split(meta.get("source_split", ""))
        if source_split == split:
            selected.append(ex)

    return selected


def load_exams_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
    mode: str = "as_is",
) -> List[Dict[str, Any]]:
    """
    Load EXAMS examples from processed JSONL.
    split:
      - "train"
      - "val"
      - "test"
    mode:
      - "as_is": use original dataset splits:
          train -> source_split == train
          val   -> source_split == validation
          test  -> source_split == test

      - "recarve": recarve the full language pool into val/test using recarve_splits.
        For split="train", returns the shuffled full pool capped by n.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported EXAMS split: {split!r}")
    if mode not in {"as_is", "recarve"}:
        raise ValueError(f"Unsupported EXAMS mode: {mode!r}")

    examples = _load_all_examples(language)
    if not examples:
        return []

    if mode == "as_is":
        selected = _filter_as_is_split(examples, split)
        dataset = _examples_to_dataset(selected).shuffle(seed=VAL_SEED)
        sampled = select_at_most(dataset, n)
        return _dataset_to_examples(sampled)

    if split == "train":
        dataset = _examples_to_dataset(examples).shuffle(seed=VAL_SEED)
        sampled = select_at_most(dataset, n)
        return _dataset_to_examples(sampled)

    n_val = n if split == "val" else None
    n_test = n if split == "test" else None
    val_ex, test_ex = load_exams_recarved(language, n_val=n_val, n_test=n_test)
    return val_ex if split == "val" else test_ex


def load_exams_recarved(
    language: str,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Recarve full EXAMS language pool into val/test examples."""
    examples = _load_all_examples(language)
    if not examples:
        return [], []

    dataset = _examples_to_dataset(examples)
    val_data, test_data = recarve_splits(
        [dataset],
        n_val,
        n_test,
        "EXAMS",
        language,
    )
    if val_data is None:
        return [], []

    return (
        _dataset_to_examples(val_data),
        _dataset_to_examples(test_data),
    )


def load_exams_splits(
    language: str,
    mode: str = "as_is",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (train_examples, val_examples)."""
    train = load_exams_examples(language, n=None, split="train", mode=mode)
    val = load_exams_examples(language, n=None, split="val", mode=mode)
    return train, val