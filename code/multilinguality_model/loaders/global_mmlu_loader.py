"""multilingual/loaders/global_mmlu_loader.py"""

from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
from multilinguality_model.utils import VAL_SEED, get_train_val_split, pick_split, select_at_most

# language code -> our language key
GLOBAL_MMLU_CONFIGS = {
    "russian": "ru",
    "chinese": "zh",
    "italian": "it",
    "spanish": "es",
    "hindi":   "hi",
}
_VALID_ANSWERS = {"A", "B", "C", "D"}


def _format_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    question = str(ex["question"]).strip()
    options = "\n".join(
        f"{letter}) {str(ex[f'option_{letter.lower()}']).strip()}"
        for letter in "ABCD"
        if ex.get(f"option_{letter.lower()}") is not None
    )
    answer = str(ex["answer"]).strip().upper()

    if not question:
        raise ValueError("Empty Global-MMLU question.")
    if answer not in _VALID_ANSWERS:
        raise ValueError(f"Invalid answer: {answer!r}")

    source_id = str(ex.get("source_id") or ex.get("sample_id") or f"global_mmlu_{language}_{idx}").strip()

    return {
        "id":        f"global_mmlu_{language}_{idx}",
        "source_id": source_id,
        "dataset":   "global_mmlu",
        "language":  language,
        "task_type": "mcq",
        "prompt":    f"{question}\n\n{options}",
        "answer":    answer,
        "metadata":  {
            "subject":                   ex.get("subject", ""),
            "subject_category":          ex.get("subject_category", ""),
            "cultural_sensitivity_label": ex.get("cultural_sensitivity_label", ""),
            "region":                    ex.get("region", ""),
            "country":                   ex.get("country", ""),
            "sample_id":                 ex.get("sample_id", ""),
        },
    }


def _is_valid(ex: Dict[str, Any]) -> bool:
    """Basic validity check — skip rows with null options or bad answers."""
    if not str(ex.get("question", "")).strip():
        return False
    # option_a and option_b must exist; c/d can be null for 2-option questions
    if not ex.get("option_a") or not ex.get("option_b"):
        return False
    answer = str(ex.get("answer", "")).strip().upper()
    if answer not in _VALID_ANSWERS:
        return False
    # If answer is C or D, the corresponding option must not be null
    if answer == "C" and not ex.get("option_c"):
        return False
    if answer == "D" and not ex.get("option_d"):
        return False
    return True


def load_global_mmlu_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
    cs_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load Global-MMLU examples.

    Args:
        cs_only: if True, only return Culturally Sensitive (CS) examples.
                 Useful if you want to prioritize culturally relevant questions.
    """
    config = GLOBAL_MMLU_CONFIGS.get(language)
    if config is None:
        print(f"[Global-MMLU] No config for {language}; skipping.")
        return []
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    # Global-MMLU has test (14k) and dev (285) splits.
    # We use test as our pool and carve train/val from it.
    # dev is too small to be useful on its own.
    ds = load_dataset("CohereLabs/Global-MMLU", config)
    split_name = "test"  # always use test as the pool
    train_pool, val_set = get_train_val_split(ds, split_name)

    data = val_set if split == "val" else train_pool

    # Filter invalid rows
    valid = [ex for ex in data if _is_valid(ex)]

    # Optional CS filter
    if cs_only:
        valid = [ex for ex in valid if ex.get("cultural_sensitivity_label") == "CS"]

    if not valid:
        print(f"[Global-MMLU] No valid examples for {language}/{split}.")
        return []

    # Respect n cap
    if n is not None:
        valid = valid[:n]

    return [_format_example(ex, language, i) for i, ex in enumerate(valid)]


def load_global_mmlu_splits(
    language: str,
    cs_only: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_global_mmlu_examples(language, n=None, split="train", cs_only=cs_only)
    val   = load_global_mmlu_examples(language, n=None, split="val",   cs_only=cs_only)
    return train, val