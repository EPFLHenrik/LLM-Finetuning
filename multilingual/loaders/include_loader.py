"""multilingual/loaders/include_loader.py

Loads INCLUDE examples for local evaluation and training.

Key design decisions:
  - prompt = raw question text only (no extra instructions injected).
    The system prompt handles all formatting guidance; this matches
    exactly what the CI eval will see in the user turn.
  - get_train_val_split() ensures we never contaminate the val set
    with training data even when only one split is available.
"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from multilingual.config import LETTERS, TARGET_LANGUAGES
from multilingual.utils import get_train_val_split, pick_split, select_at_most


def _build_raw_question(ex: Dict[str, Any]) -> str:
    """
    Format the INCLUDE row into a plain question string.

    This is the user-turn content — no boxed-format instructions,
    no system-prompt material. Keep it clean.
    """
    question = str(ex["question"])
    choices = [
        str(ex["option_a"]),
        str(ex["option_b"]),
        str(ex["option_c"]),
        str(ex["option_d"]),
    ]
    options = "\n".join(f"{LETTERS[i]}) {choice}" for i, choice in enumerate(choices))
    return f"{question}\n\n{options}"


def _format_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    return {
        "id": f"include_{language}_{idx}",
        "dataset": "include",
        "language": language,
        "task_type": "mcq",
        "prompt": _build_raw_question(ex),       # raw question, no instructions
        "answer": LETTERS[int(ex["answer"])],
        "metadata": {
            "language":         ex.get("language"),
            "country":          ex.get("country"),
            "domain":           ex.get("domain"),
            "subject":          ex.get("subject"),
            "regional_feature": ex.get("regional_feature"),
            "level":            ex.get("level"),
        },
    }


def load_include_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",          # "val" or "train"
) -> List[Dict[str, Any]]:
    """
    Load INCLUDE examples for a given language.

    Args:
        language:  Key from TARGET_LANGUAGES.
        n:         Max examples to return (None = all).
        split:     'val'   → the held-out validation slice (never train on this)
                   'train' → the training pool
    """
    include_config = TARGET_LANGUAGES[language]["include_config"]
    if include_config is None:
        print(f"[INCLUDE] No config for {language}; skipping.")
        return []

    ds = load_dataset("CohereLabs/include-base-44", include_config)
    split_name = pick_split(ds)
    train_pool, val_set = get_train_val_split(ds, split_name)

    data = val_set if split == "val" else train_pool

    if len(data) == 0:
        print(f"[INCLUDE] Empty {split} split for {language}.")
        return []

    sampled = select_at_most(data, n)
    return [_format_example(sampled[i], language, i) for i in range(len(sampled))]


def load_include_splits(
    language: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convenience: return (train_examples, val_examples) in one call.
    Useful for training scripts that need both.
    """
    train = load_include_examples(language, n=None, split="train")
    val   = load_include_examples(language, n=None, split="val")
    return train, val