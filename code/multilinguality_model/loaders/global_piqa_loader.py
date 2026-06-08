"""multilingual/loaders/global_piqa_loader.py

Loads Global-PIQA examples.

Data source priority:
  1. Preprocessed disk file if available:
       data/multilinguality/processed/global_piqa/global_piqa_{language}_all.jsonl
  2. HuggingFace fallback:
       mrlbenchmarks/global-piqa-nonparallel

Modes:
  as_is:
      legacy carve using get_train_val_split:
        train/test pool = 85%
        val = 15%

  recarve:
      recarve full pool into train/val/test using recarve_splits.
      If only val/test are requested, behaves like the previous eval setup.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from multilinguality_model.loaders.recarve_utils import recarve_splits
from multilinguality_model.utils import VAL_SEED, get_train_val_split, select_at_most


GLOBAL_PIQA_DISK_DIR = Path("data/multilinguality/processed/global_piqa")
GLOBAL_PIQA_CONFIGS = {
    "italian": ["ita_latn"],
    "spanish": ["spa_latn_mexi", "spa_latn_peru", "spa_latn_spai"],
    "hindi": ["hin_deva", "hin_latn"],
    "chinese": ["cmn_hans", "cmn_hant"],
    "russian": ["rus_cyrl"],
}
_LABEL_TO_LETTER = {0: "A", 1: "B"}


def _parse_supplement(x: Any) -> Dict[str, Any]:
    if not x or isinstance(x, float):
        return {}
    try:
        return json.loads(str(x))
    except (json.JSONDecodeError, ValueError):
        return {}


def _format_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    prompt = str(ex["prompt"]).strip()
    sol0 = str(ex["solution0"]).strip()
    sol1 = str(ex["solution1"]).strip()
    label = int(ex["label"])
    if not prompt or not sol0 or not sol1:
        raise ValueError("Empty prompt or solution.")
    if label not in _LABEL_TO_LETTER:
        raise ValueError(f"Unexpected label: {label!r}")
    supp = _parse_supplement(ex.get("supplement", {}))
    source_id = str(
        ex.get("source_id")
        or ex.get("id")
        or ex.get("example_id")
        or f"global_piqa_{language}_{idx}"
    ).strip()

    return {
        "id": f"global_piqa_{language}_{idx}",
        "source_id": source_id,
        "dataset": "global_piqa",
        "language": language,
        "task_type": "mcq",
        "prompt": f"{prompt}\n\nA) {sol0}\nB) {sol1}",
        "answer": _LABEL_TO_LETTER[label],
        "metadata": {
            "config": ex.get("language", ex.get("config_name", "")),
            "approx_cultural_score": ex.get("approx_cultural_score", ""),
            "example_id": ex.get("example_id", ""),
            "source_id": source_id,
            "source_split": ex.get("source_split", "test"),
            "category": supp.get("category", ""),
        },
    }


def _validate_disk_example(ex: Dict[str, Any]) -> bool:
    required = ["id", "dataset", "language", "task_type", "prompt", "answer", "metadata"]
    for key in required:
        if key not in ex:
            return False
    if ex["dataset"] != "global_piqa":
        return False
    if ex["task_type"] != "mcq":
        return False
    if ex["answer"] not in {"A", "B"}:
        return False
    if not str(ex["prompt"]).strip():
        return False
    if "A)" not in ex["prompt"] or "B)" not in ex["prompt"]:
        return False
    if not isinstance(ex["metadata"], dict):
        return False
    return True


def _load_disk_pool(language: str) -> Optional[List[Dict[str, Any]]]:
    path = GLOBAL_PIQA_DISK_DIR / f"global_piqa_{language}_all.jsonl"
    if not path.exists():
        return None
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
            if not _validate_disk_example(ex):
                skipped += 1
                continue

            ex["language"] = language
            if "metadata" not in ex or not isinstance(ex["metadata"], dict):
                ex["metadata"] = {}

            source_id = str(
                ex.get("source_id")
                or ex.get("id")
                or ex["metadata"].get("source_id")
                or ex["metadata"].get("example_id")
                or ""
            ).strip()

            if not source_id:
                source_id = f"global_piqa_{language}_{len(examples)}"

            ex["source_id"] = source_id
            ex["metadata"].setdefault("source_id", source_id)

            examples.append(ex)
    if skipped:
        print(f"[Global-PIQA] Skipped {skipped} invalid disk rows for {language}.")
    if not examples:
        print(f"[Global-PIQA] Disk file exists but no valid rows for {language}: {path}")
        return None
    print(f"[Global-PIQA] Loaded {len(examples)} processed examples from disk for {language}.")
    return examples


def _load_hf_pool(language: str):
    configs = GLOBAL_PIQA_CONFIGS.get(language)
    if not configs:
        print(f"[Global-PIQA] No config for {language}; skipping.")
        return None
    splits = []
    for config in configs:
        try:
            ds = load_dataset("mrlbenchmarks/global-piqa-nonparallel", config)
            test_ds = ds["test"]
            test_ds = test_ds.add_column("config_name", [config] * len(test_ds))
            test_ds = test_ds.add_column("source_split", ["test"] * len(test_ds))
            splits.append(test_ds)
        except Exception as e:
            print(f"[Global-PIQA] Failed to load config {config}: {e}")
    if not splits:
        print(f"[Global-PIQA] No data loaded for {language}.")
        return None
    return concatenate_datasets(splits).shuffle(seed=VAL_SEED)


def _examples_from_dataset(data, language: str) -> List[Dict[str, Any]]:
    examples, skipped = [], 0
    for i in range(len(data)):
        try:
            examples.append(_format_example(data[i], language, i))
        except ValueError:
            skipped += 1
    if skipped:
        print(f"[Global-PIQA] Skipped {skipped} invalid HF rows for {language}.")
    return examples


def _load_example_pool(language: str) -> List[Dict[str, Any]]:
    disk_examples = _load_disk_pool(language)
    if disk_examples is not None:
        return disk_examples
    combined = _load_hf_pool(language)
    if combined is None:
        return []
    return _examples_from_dataset(combined, language)


def _examples_to_dataset(examples: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(examples)


def _dataset_to_examples(data) -> List[Dict[str, Any]]:
    return [dict(row) for row in data]


def _split_train_val_test_as_is(
    examples: List[Dict[str, Any]],
    n: Optional[int],
    split: str,
) -> List[Dict[str, Any]]:
    dataset = _examples_to_dataset(examples)
    ds_dict = DatasetDict({"test": dataset})
    train_pool, val_set = get_train_val_split(ds_dict, "test")
    if split == "val":
        return _dataset_to_examples(select_at_most(val_set, n))
    if split in {"train", "test"}:
        return _dataset_to_examples(select_at_most(train_pool, n))
    raise ValueError(f"Unsupported split for Global-PIQA: {split!r}")


def load_global_piqa_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
    mode: str = "recarve",
) -> List[Dict[str, Any]]:
    """
    Load Global-PIQA examples for one split.
    split:
      - "train"
      - "val"
      - "test"
    mode:
      - "recarve": use recarve_splits for val/test; use shuffled full pool for train.
      - "as_is": use legacy get_train_val_split carve.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split for Global-PIQA: {split!r}")

    examples = _load_example_pool(language)
    if not examples:
        return []
    if mode == "recarve":
        if split == "train":
            dataset = _examples_to_dataset(examples).shuffle(seed=VAL_SEED)
            sampled = select_at_most(dataset, n)
            return _dataset_to_examples(sampled)

        n_val = n if split == "val" else None
        n_test = n if split == "test" else None
        val_ex, test_ex = load_global_piqa_recarved(
            language,
            n_val=n_val,
            n_test=n_test,
        )
        return val_ex if split == "val" else test_ex
    
    if mode == "as_is":
        return _split_train_val_test_as_is(examples, n, split)
    raise ValueError(f"Unsupported Global-PIQA mode: {mode!r}")


def load_global_piqa_recarved(
    language: str,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Recarve full example pool into val/test examples using the existing helper."""
    examples = _load_example_pool(language)
    if not examples:
        return [], []

    dataset = _examples_to_dataset(examples)
    val_data, test_data = recarve_splits(
        [dataset],
        n_val,
        n_test,
        "Global-PIQA",
        language,
    )
    if val_data is None:
        return [], []
    return _dataset_to_examples(val_data),_dataset_to_examples(test_data)