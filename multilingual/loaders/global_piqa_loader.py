"""multilingual/loaders/global_piqa_loader.py

Loads Global-PIQA examples for eval only (never training).

HF only has a 'test' split. Two modes:
  as_is  : val = carved 15%, test = remaining 85% (legacy behaviour).
  recarve: merge all regional configs, recarve into val/test via split_caps
           (default 50/50 per language).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from datasets import concatenate_datasets, load_dataset

from multilingual.loaders.recarve_utils import recarve_splits
from multilingual.utils import VAL_SEED, get_train_val_split, select_at_most

GLOBAL_PIQA_CONFIGS = {
    "italian": ["ita_latn"],
    "spanish": ["spa_latn_mexi", "spa_latn_peru", "spa_latn_spai"],
    "hindi":   ["hin_deva"],
    "chinese": ["cmn_hans"],
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
    sol0   = str(ex["solution0"]).strip()
    sol1   = str(ex["solution1"]).strip()
    label  = int(ex["label"])

    if not prompt or not sol0 or not sol1:
        raise ValueError("Empty prompt or solution.")
    if label not in _LABEL_TO_LETTER:
        raise ValueError(f"Unexpected label: {label!r}")

    supp = _parse_supplement(ex.get("supplement", {}))
    return {
        "id":        f"global_piqa_{language}_{idx}",
        "dataset":   "global_piqa",
        "language":  language,
        "task_type": "mcq",
        "prompt":    f"{prompt}\n\nA) {sol0}\nB) {sol1}",
        "answer":    _LABEL_TO_LETTER[label],
        "metadata": {
            "config":                ex.get("language", ""),
            "approx_cultural_score": ex.get("approx_cultural_score", ""),
            "example_id":            ex.get("example_id", ""),
            "category":              supp.get("category", ""),
        },
    }


def _load_hf_pool(language: str):
    """Concatenate all regional HF configs for a language (test split only)."""
    configs = GLOBAL_PIQA_CONFIGS.get(language)
    if not configs:
        print(f"[Global-PIQA] No config for {language}; skipping.")
        return None

    splits = []
    for config in configs:
        try:
            ds = load_dataset("mrlbenchmarks/global-piqa-nonparallel", config)
            splits.append(ds["test"])
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
        print(f"[Global-PIQA] Skipped {skipped} invalid rows for {language}.")
    return examples


def load_global_piqa_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
    mode: str = "recarve",
) -> List[Dict[str, Any]]:
    """
    Load Global-PIQA examples for a single split (eval only, never train).

    mode='recarve' (default): recarve HF test into val/test (50/50 default).
    mode='as_is':             use get_train_val_split legacy 85/15 carve.
    """
    if split == "train":
        return []  # never train on Global-PIQA

    combined = _load_hf_pool(language)
    if combined is None:
        return []

    if mode == "recarve":
        n_val  = n if split == "val"  else None
        n_test = n if split == "test" else None
        val_ex, test_ex = load_global_piqa_recarved(language, n_val=n_val, n_test=n_test)
        return val_ex if split == "val" else test_ex

    # as_is: legacy 85/15 carve via get_train_val_split
    from datasets import DatasetDict
    ds_dict = DatasetDict({"test": combined})
    train_pool, val_set = get_train_val_split(ds_dict, "test")
    data    = val_set if split == "val" else train_pool
    sampled = select_at_most(data, n)
    return _examples_from_dataset(sampled, language)


def load_global_piqa_recarved(
    language: str,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Recarve HF test pool into (val_examples, test_examples). Default 50/50."""
    combined = _load_hf_pool(language)
    if combined is None:
        return [], []

    val_data, test_data = recarve_splits(
        [combined], n_val, n_test, "Global-PIQA", language
    )
    if val_data is None:
        return [], []

    return (
        _examples_from_dataset(val_data,  language),
        _examples_from_dataset(test_data, language),
    )