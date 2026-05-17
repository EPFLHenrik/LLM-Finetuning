"""multilingual/loaders/global_piqa_loader.py"""

import json
from typing import Any, Dict, List, Optional, Tuple

from datasets import concatenate_datasets, load_dataset

from multilingual.utils import VAL_SEED, get_train_val_split, select_at_most

# Each language may have multiple regional configs.
# All configs for a language are concatenated before splitting.
GLOBAL_PIQA_CONFIGS = {
    "italian": ["ita_latn"],
    "spanish": ["spa_latn_mexi", "spa_latn_peru", "spa_latn_spai"],
    "hindi":   ["hin_deva"],
    "chinese": ["cmn_hans"],
    "russian": ["rus_cyrl"],
}

_LABEL_TO_LETTER = {0: "A", 1: "B"}


def _parse_supplement(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if not x or (isinstance(x, float)):
        return {}
    try:
        return json.loads(str(x))
    except (json.JSONDecodeError, ValueError):
        return {}


def _format_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    prompt    = str(ex["prompt"]).strip()
    sol0      = str(ex["solution0"]).strip()
    sol1      = str(ex["solution1"]).strip()
    label     = int(ex["label"])

    if not prompt:
        raise ValueError("Empty Global-PIQA prompt.")
    if label not in _LABEL_TO_LETTER:
        raise ValueError(f"Unexpected label: {label!r}")
    if not sol0 or not sol1:
        raise ValueError("Empty solution.")

    answer = _LABEL_TO_LETTER[label]
    supp   = _parse_supplement(ex.get("supplement", {}))

    return {
        "id":        f"global_piqa_{language}_{idx}",
        "dataset":   "global_piqa",
        "language":  language,
        "task_type": "mcq",
        "prompt":    f"{prompt}\n\nA) {sol0}\nB) {sol1}",
        "answer":    answer,
        "metadata":  {
            "config":                ex.get("language", ""),
            "approx_cultural_score": ex.get("approx_cultural_score", ""),
            "example_id":            ex.get("example_id", ""),
            "category":              supp.get("category", ""),
        },
    }


def load_global_piqa_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
) -> List[Dict[str, Any]]:
    """
    Load Global-PIQA examples. split='val' or 'train'.

    Each language has 1-3 regional configs of 100 rows each (test split only).
    All configs are concatenated before the 85/15 train/val carve-out.
    """
    configs = GLOBAL_PIQA_CONFIGS.get(language)
    if not configs:
        print(f"[Global-PIQA] No config for {language}; skipping.")
        return []
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    # Load and concatenate all regional configs
    splits = []
    for config in configs:
        try:
            ds = load_dataset("mrlbenchmarks/global-piqa-nonparallel", config)
            splits.append(ds["test"])
        except Exception as e:
            print(f"[Global-PIQA] Failed to load config {config}: {e}")

    if not splits:
        print(f"[Global-PIQA] No data loaded for {language}.")
        return []

    combined = concatenate_datasets(splits).shuffle(seed=VAL_SEED)

    # Wrap in a DatasetDict-like object for get_train_val_split
    from datasets import DatasetDict
    ds_dict = DatasetDict({"test": combined})
    train_pool, val_set = get_train_val_split(ds_dict, "test")

    data    = val_set if split == "val" else train_pool
    sampled = select_at_most(data, n)

    examples, skipped = [], 0
    for i in range(len(sampled)):
        try:
            examples.append(_format_example(sampled[i], language, i))
        except ValueError:
            skipped += 1

    if skipped:
        print(f"[Global-PIQA] Skipped {skipped} invalid rows for {language}/{split}.")
    return examples


def load_global_piqa_splits(
    language: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_global_piqa_examples(language, n=None, split="train")
    val   = load_global_piqa_examples(language, n=None, split="val")
    return train, val