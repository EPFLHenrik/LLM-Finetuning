"""multilingual/mgsm_loader.py

Loads MGSM examples for local evaluation and training.

MGSM only has a 'test' split (250 examples per language), so we apply
the same train/val carve-out logic as the INCLUDE loader.
"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset

from multilingual.config import TARGET_LANGUAGES
from multilingual.utils import get_train_val_split, pick_split, select_at_most


def _infer_key(ex: Dict[str, Any], candidates: List[str]) -> str:
    for key in candidates:
        if key in ex:
            return key
    raise KeyError(f"None of {candidates} found in example keys: {list(ex.keys())}")


def _normalize_gold(answer: Any) -> str:
    s = str(answer).strip()
    if "####" in s:
        s = s.split("####")[-1].strip()
    return s


def _build_raw_question(ex: Dict[str, Any]) -> str:
    """
    Return the raw question string for the user turn.
    No extra instructions — those live in the system prompt.
    """
    question_key = _infer_key(ex, ["question", "problem", "input", "query", "prompt"])
    return str(ex[question_key])


def _format_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    question_key = _infer_key(ex, ["question", "problem", "input", "query", "prompt"])
    answer_key   = _infer_key(ex, ["answer_number", "answer", "target", "output", "label"])

    return {
        "id":        f"mgsm_{language}_{idx}",
        "dataset":   "mgsm",
        "language":  language,
        "task_type": "math",
        "prompt":    _build_raw_question(ex),
        "answer":    _normalize_gold(ex[answer_key]),
        "metadata":  {k: ex[k] for k in ex.keys() if k not in {question_key, answer_key}},
    }


def _resolve_config(language: str) -> Optional[str]:
    mgsm_configs = TARGET_LANGUAGES[language]["mgsm_configs"]
    if not mgsm_configs:
        return None

    available = get_dataset_config_names("juletxara/mgsm")
    lookup = {c.lower(): c for c in available}
    for candidate in mgsm_configs:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_mgsm_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
) -> List[Dict[str, Any]]:
    """
    Load MGSM examples for a given language.

    Args:
        language: Key from TARGET_LANGUAGES.
        n:        Max examples (None = all).
        split:    'val' or 'train' — same carve-out logic as INCLUDE.
    """
    config = _resolve_config(language)
    if config is None:
        print(f"[MGSM] No config for {language}; skipping.")
        return []

    ds = load_dataset("juletxara/mgsm", config)
    split_name = pick_split(ds)
    train_pool, val_set = get_train_val_split(ds, split_name)

    data = val_set if split == "val" else train_pool

    if len(data) == 0:
        print(f"[MGSM] Empty {split} split for {language}.")
        return []

    sampled = select_at_most(data, n)
    return [_format_example(sampled[i], language, i) for i in range(len(sampled))]


def load_mgsm_splits(
    language: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples) in one call."""
    train = load_mgsm_examples(language, n=None, split="train")
    val   = load_mgsm_examples(language, n=None, split="val")
    return train, val