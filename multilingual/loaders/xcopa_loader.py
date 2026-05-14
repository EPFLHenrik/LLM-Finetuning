"""multilingual/loaders/xcopa_loader.py"""

from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
from multilingual.utils import VAL_SEED, select_at_most

XCOPA_CONFIGS = {
    "italian": "it",
    "chinese": "zh",
}

_QUESTION_TEMPLATES = {
    "italian": {
        "cause":  "Qual è stata la causa?",
        "effect": "Che cosa è successo come risultato?",
    },
    "chinese": {
        "cause":  "原因是什么？",
        "effect": "结果发生了什么？",
    },
}


def _format_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    premise  = str(ex["premise"]).strip()
    q_type   = str(ex["question"]).strip()
    if q_type not in _QUESTION_TEMPLATES[language]:
        raise ValueError(f"Unexpected XCOPA question type: {q_type!r}")
    q_line   = _QUESTION_TEMPLATES[language][q_type]
    choice1  = str(ex["choice1"]).strip()
    choice2  = str(ex["choice2"]).strip()
    prompt   = f"{premise}\n\n{q_line}\n\nA) {choice1}\nB) {choice2}"

    label = int(ex["label"])
    if label not in {0, 1}:
        raise ValueError(f"Unexpected XCOPA label: {label!r}")
    answer = "A" if label == 0 else "B"

    return {
        "id":        f"xcopa_{language}_{idx}",
        "dataset":   "xcopa",
        "language":  language,
        "task_type": "mcq",
        "prompt":    prompt,
        "answer":    answer,
        "metadata":  {
            "question_type": q_type,
            "changed":       bool(ex.get("changed", False)),
            "idx":           int(ex.get("idx", idx)),
        },
    }


def load_xcopa_examples(
    language: str,
    n: Optional[int] = None,
    split: str = "val",
) -> List[Dict[str, Any]]:
    """
    Load XCOPA examples. split='val' or 'train'.
    Uses HF 'validation' (100) as local val and 'test' (500) as train pool.
    """
    config = XCOPA_CONFIGS.get(language)
    if config is None:
        print(f"[XCOPA] No config for {language} (only Italian and Chinese available); skipping.")
        return []
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    ds = load_dataset("cambridgeltl/xcopa", config)
    hf_split = "validation" if split == "val" else "test"
    data = ds[hf_split].shuffle(seed=VAL_SEED)

    sampled = select_at_most(data, n)
    return [_format_example(sampled[i], language, i) for i in range(len(sampled))]


def load_xcopa_splits(
    language: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_xcopa_examples(language, n=None, split="train")
    val   = load_xcopa_examples(language, n=None, split="val")
    return train, val