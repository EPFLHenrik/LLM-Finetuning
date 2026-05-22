"""multilingual/loaders/belebele_loader.py"""

from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
from multilingual.utils import VAL_SEED, get_train_val_split, pick_split, select_at_most

BELEBELE_CONFIGS = {
    "russian": "rus_Cyrl",
    "chinese": "zho_Hans",
    "italian": "ita_Latn",
    "spanish": "spa_Latn",
    "hindi":   "hin_Deva",
}
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}


def _format_example(ex: Dict[str, Any], language: str, split: str, idx: int) -> Dict[str, Any]:
    passage  = str(ex["flores_passage"]).strip()
    question = str(ex["question"]).strip()
    options  = "\n".join(
        f"{letter}) {str(ex[f'mc_answer{num}']).strip()}"
        for num, letter in _NUM_TO_LETTER.items()
    )
    prompt = f"{passage}\n\n{question}\n\n{options}"

    answer_num = str(ex["correct_answer_num"]).strip()
    if answer_num not in _NUM_TO_LETTER:
        raise ValueError(f"Unexpected Belebele answer: {answer_num!r}")
    answer = _NUM_TO_LETTER[answer_num]

    return {
        "id":        f"belebele_{language}_{split}_{idx}",
        "dataset":   "belebele",
        "language":  language,
        "task_type": "mcq",
        "prompt":    prompt,
        "answer":    answer,
        "metadata":  {
            "config":  BELEBELE_CONFIGS[language],
            "dialect": ex.get("dialect", ""),
            "link":    ex.get("link", ""),
        },
    }


def load_belebele_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
) -> List[Dict[str, Any]]:
    """Load Belebele examples. split='val' or 'train'."""
    config = BELEBELE_CONFIGS.get(language)
    if config is None:
        print(f"[Belebele] No config for {language}; skipping.")
        return []
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    ds = load_dataset("facebook/belebele", config)
    split_name = pick_split(ds)
    train_pool, val_set = get_train_val_split(ds, split_name)

    data = val_set if split == "val" else train_pool
    sampled = select_at_most(data, n)
    return [_format_example(sampled[i], language, split, i) for i in range(len(sampled))]


def load_belebele_splits(
    language: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_belebele_examples(language, n=None, split="train")
    val   = load_belebele_examples(language, n=None, split="val")
    return train, val