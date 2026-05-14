"""multilingual/loaders/milu_loader.py"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from multilingual.utils import VAL_SEED, select_at_most

# MILU answer is "option1".."option4" — map to A/B/C/D
_TARGET_TO_LETTER = {"option1": "A", "option2": "B", "option3": "C", "option4": "D"}
_OPTION_KEYS = ["option1", "option2", "option3", "option4"]


def _format_example(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = str(ex["question"])
    options = "\n".join(
        f"{letter}) {ex[key]}"
        for key, letter in zip(_OPTION_KEYS, "ABCD")
    )
    return {
        "id":        f"milu_hindi_{idx}",
        "dataset":   "milu",
        "language":  "hindi",
        "task_type": "mcq",
        "prompt":    f"{question}\n\n{options}",
        "answer":    _TARGET_TO_LETTER[str(ex["target"])],
        "metadata":  {
            "subject":       ex.get("subject", ""),
            "domain":        ex.get("domain", ""),
            "is_translated": ex.get("is_translated", False),
        },
    }


def load_milu_examples(
    n: Optional[int] = None,
    split: str = "val",
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load MILU Hindi examples.

    MILU has a real validation split (8,933 samples), so:
      split='val'   → loads the 'validation' split  (never train on this)
      split='train' → loads the 'test' split, shuffled, used as training pool
                      (the official 'test' split is ~14k Hindi examples)
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    hf_split = "validation" if split == "val" else "test"
    data = load_dataset(
        "ai4bharat/MILU",
        data_dir="Hindi",
        split=hf_split,
        token=hf_token or True,
    )

    data = data.shuffle(seed=VAL_SEED)
    sampled = select_at_most(data, n)
    return [_format_example(sampled[i], i) for i in range(len(sampled))]


def load_milu_splits(
    hf_token: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_milu_examples(n=None, split="train", hf_token=hf_token)
    val   = load_milu_examples(n=None, split="val",   hf_token=hf_token)
    return train, val