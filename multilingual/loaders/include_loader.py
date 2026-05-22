"""multilingual/loaders/include_loader.py

Loads INCLUDE examples for eval (val/test). Never used for training.

HF splits available: 'validation', 'test' (no train split).

Two modes:
  as_is   (default): 'validation' → val, 'test' → test, used directly.
  recarve:           merge validation+test, recarve into val/test
                     with sizes from split_caps (default 50/50).
"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset

from multilingual.config import LETTERS, TARGET_LANGUAGES
from multilingual.loaders.recarve_utils import recarve_splits
from multilingual.utils import VAL_SEED, select_at_most

# Ordered list: first match per internal key wins
_HF_SPLIT_MAP = [
    ("train",      "train"),
    ("validation", "val"),
    ("val",        "val"),
    ("test",       "test"),
    ("dev",        "val"),
]


def _get_hf_splits(ds: DatasetDict) -> Dict[str, Any]:
    """Map available HF splits to internal keys. First match per key wins."""
    result: Dict[str, Any] = {}
    for hf_name, internal in _HF_SPLIT_MAP:
        if hf_name in ds and internal not in result:
            result[internal] = ds[hf_name]
    return result


def _build_prompt(ex: Dict[str, Any]) -> str:
    question = str(ex["question"])
    choices  = [str(ex[f"option_{l.lower()}"]) for l in LETTERS[:4]]
    options  = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{options}"


def _format_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    return {
        "id":        f"include_{language}_{idx}",
        "dataset":   "include",
        "language":  language,
        "task_type": "mcq",
        "prompt":    _build_prompt(ex),
        "answer":    LETTERS[int(ex["answer"])],
        "metadata": {
            "language":         ex.get("language"),
            "country":          ex.get("country"),
            "domain":           ex.get("domain"),
            "subject":          ex.get("subject"),
            "regional_feature": ex.get("regional_feature"),
            "level":            ex.get("level"),
        },
    }


def _load_hf(language: str) -> Optional[Dict[str, Any]]:
    """Load HF dataset and return mapped splits dict, or None if no config."""
    include_config = TARGET_LANGUAGES[language]["include_config"]
    if include_config is None:
        print(f"[INCLUDE] No config for {language}; skipping.")
        return None
    ds = load_dataset("CohereLabs/include-base-44", include_config)
    return _get_hf_splits(ds)


def load_include_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
    mode: str = "as_is",
) -> List[Dict[str, Any]]:
    """
    Load INCLUDE examples for a single split.

    For recarve mode with both val+test needed, prefer load_include_recarved()
    to avoid a double HF download.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
    if split == "train":
        return []  # never train on INCLUDE

    if mode == "recarve":
        n_val  = n if split == "val"  else None
        n_test = n if split == "test" else None
        val_ex, test_ex = load_include_recarved(language, n_val=n_val, n_test=n_test)
        return val_ex if split == "val" else test_ex

    # as_is: use HF splits directly
    splits = _load_hf(language)
    if splits is None:
        return []
    if split not in splits:
        print(f"[INCLUDE] No '{split}' split for {language} "
              f"(available: {list(splits)}); skipping.")
        return []

    data    = splits[split].shuffle(seed=VAL_SEED)
    sampled = select_at_most(data, n)
    return [_format_example(sampled[i], language, i) for i in range(len(sampled))]


def load_include_recarved(
    language: str,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge HF validation+test and recarve into (val_examples, test_examples)."""
    splits = _load_hf(language)
    if splits is None:
        return [], []

    to_merge = [splits[k] for k in ("val", "test") if k in splits]
    val_data, test_data = recarve_splits(to_merge, n_val, n_test, "INCLUDE", language)

    if val_data is None or test_data is None:
        return [], []

    val_ex  = [_format_example(val_data[i],  language, i) for i in range(len(val_data))]
    test_ex = [_format_example(test_data[i], language, i) for i in range(len(test_data))]
    return val_ex, test_ex