"""multilingual/loaders/ceval_loader.py"""

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import random
from datasets import get_dataset_config_names, load_dataset
from multilingual.utils import VAL_SEED


# C-Eval: one config per subject, 52 subjects total.
# "val" has labels; "test" has no labels and should not be used.
# "dev" has very few examples per subject and is skipped.


def _get_all_subjects() -> List[str]:
    return get_dataset_config_names("ceval/ceval-exam")


def _format_example(ex: Dict[str, Any], subject: str, split: str, idx: int) -> Dict[str, Any]:
    question = str(ex["question"]).strip()
    options = "\n".join(f"{letter}) {str(ex[letter]).strip()}" for letter in ("A", "B", "C", "D"))
    answer = str(ex["answer"]).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unexpected C-Eval answer: {answer!r}")

    return {
        "id": f"ceval_chinese_{split}_{subject}_{idx}",
        "dataset": "ceval",
        "language": "chinese",
        "split": split,
        "task_type": "mcq",
        "prompt": f"{question}\n\n{options}",
        "answer": answer,
        "metadata": {"subject": subject, "explanation": ex.get("explanation", "")},
    }


def _load_subject_val(subject: str) -> List[Dict[str, Any]]:
    """Load the labeled validation split for one subject."""
    try:
        ds = load_dataset("ceval/ceval-exam", name=subject)
    except Exception as e:
        print(f"[C-Eval] Failed to load subject {subject!r}: {e}")
        return []

    if "val" not in ds or len(ds["val"]) == 0:
        return []

    return list(ds["val"])


def _load_all_labeled_rows() -> Tuple[List[Dict[str, Any]], List[str]]:
    all_rows = []
    subject_labels = []
    for subject in _get_all_subjects():
        rows = _load_subject_val(subject)
        all_rows.extend(rows)
        subject_labels.extend([subject] * len(rows))

    return all_rows, subject_labels


def _stratified_split(all_rows: List[Dict[str, Any]], subject_labels: List[str], val_fraction: float, seed: int) -> Tuple[List[Tuple[Dict[str, Any], str]], List[Tuple[Dict[str, Any], str]]]:
    rng = random.Random(seed)
    by_subject = defaultdict(list)
    for row, subject in zip(all_rows, subject_labels):
        by_subject[subject].append((row, subject))
    train_pairs = []
    val_pairs = []
    for pairs in by_subject.values():
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_pairs.extend(shuffled[:n_val])
        train_pairs.extend(shuffled[n_val:])

    return train_pairs, val_pairs


def _load_ceval_pairs(val_fraction: float) -> Tuple[List[Tuple[Dict[str, Any], str]], List[Tuple[Dict[str, Any], str]]]:
    all_rows, subject_labels = _load_all_labeled_rows()
    return _stratified_split(all_rows, subject_labels, val_fraction, VAL_SEED)


def load_ceval_examples(n: Optional[int] = None, split: str = "val", val_fraction: float = 0.20) -> List[Dict[str, Any]]:
    """
    Load C-Eval examples for Chinese.
    split="train" returns the training pool.
    split="val" returns the validation pool.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    train_pairs, val_pairs = _load_ceval_pairs(val_fraction)
    pairs = val_pairs if split == "val" else train_pairs

    rng = random.Random(VAL_SEED)
    pairs = pairs[:]
    rng.shuffle(pairs)

    if n is not None:
        pairs = pairs[:n]

    return [_format_example(row, subject, split, idx) for idx, (row, subject) in enumerate(pairs)]


def load_ceval_splits(val_fraction: float = 0.20) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples), loading HF only once."""
    train_pairs, val_pairs = _load_ceval_pairs(val_fraction)
    rng = random.Random(VAL_SEED)
    train_pairs = train_pairs[:]
    val_pairs = val_pairs[:]
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    train = [_format_example(row, subject, "train", idx) for idx, (row, subject) in enumerate(train_pairs)]
    val = [_format_example(row, subject, "val", idx) for idx, (row, subject) in enumerate(val_pairs)]
    return train, val