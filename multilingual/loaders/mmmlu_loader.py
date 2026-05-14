"""multilingual/mmmlu_loader.py"""

from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
from multilingual.config import TARGET_LANGUAGES
from multilingual.utils import VAL_SEED, select_at_most


# map our language keys to MMMLU config names.
# Russian is absent from MMMLU.
MMMLU_CONFIGS = {
    "spanish": "ES_LA",
    "hindi":   "HI_IN",
    "italian": "IT_IT",
    "chinese": "ZH_CN",
}

N_VAL_PER_SUBJECT = 4   # ~4 * 57 subjects ≈ 228 val examples per language


def _format_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    question = str(ex["Question"])
    options = "\n".join(f"{letter}) {ex[letter]}" for letter in ("A", "B", "C", "D"))
    return {
        "id":        f"mmmlu_{language}_{idx}",
        "dataset":   "mmmlu",
        "language":  language,
        "task_type": "mcq",
        "prompt":    f"{question}\n\n{options}",
        "answer":    str(ex["Answer"]).strip().upper(),
        "metadata":  {"subject": ex.get("Subject", "")},
    }


def _stratified_split(data, n_val_per_subject: int, seed: int):
    """
    Split by Subject so val coverage is uniform across all 57 categories.
    Returns (train_indices, val_indices).
    """
    from collections import defaultdict
    import random

    rng = random.Random(seed)

    by_subject: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(data):
        by_subject[row["Subject"]].append(i)

    val_idx, train_idx = [], []
    for indices in by_subject.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n = min(n_val_per_subject, len(shuffled))
        val_idx.extend(shuffled[:n])
        train_idx.extend(shuffled[n:])

    return train_idx, val_idx


def _load_split(language: str, split: str) -> Tuple[List[int], Any]:
    config = MMMLU_CONFIGS.get(language)
    if config is None:
        return [], None

    data = load_dataset("openai/MMMLU", config, split="test")
    train_idx, val_idx = _stratified_split(data, N_VAL_PER_SUBJECT, VAL_SEED)
    indices = val_idx if split == "val" else train_idx
    return indices, data


def load_mmmlu_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
) -> List[Dict[str, Any]]:
    """Load MMMLU examples. split='val' or 'train'."""
    if language not in MMMLU_CONFIGS:
        print(f"[MMMLU] No config for {language} (Russian not in MMMLU); skipping.")
        return []

    indices, data = _load_split(language, split)
    if not indices:
        return []

    # Respect n cap
    if n is not None:
        indices = indices[:n]

    return [_format_example(data[i], language, idx) for idx, i in enumerate(indices)]


def load_mmmlu_splits(language: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_mmmlu_examples(language, n=None, split="train")
    val   = load_mmmlu_examples(language, n=None, split="val")
    return train, val