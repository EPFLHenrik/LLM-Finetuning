"""multilingual/ceval_loader.py"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset

from multilingual.utils import VAL_SEED, select_at_most

# C-Eval: one config per subject, 52 subjects total.
# 'val' split has labels (~1346 total); 'test' has NO labels — never use test.
# 'dev' split (5 examples per subject) is few-shot only, skip for eval/train.


def _get_all_subjects() -> List[str]:
    return get_dataset_config_names("ceval/ceval-exam")


def _format_example(ex: Dict[str, Any], subject: str, idx: int) -> Dict[str, Any]:
    question = str(ex["question"])
    options = "\n".join(f"{letter}) {ex[letter]}" for letter in ("A", "B", "C", "D"))
    return {
        "id":        f"ceval_chinese_{idx}",
        "dataset":   "ceval",
        "language":  "chinese",
        "task_type": "mcq",
        "prompt":    f"{question}\n\n{options}",
        "answer":    str(ex["answer"]).strip().upper(),
        "metadata":  {"subject": subject, "explanation": ex.get("explanation", "")},
    }


def _load_subject_val(subject: str) -> List[Dict[str, Any]]:
    """Load the val split for one subject (has labels)."""
    try:
        ds = load_dataset("ceval/ceval-exam", name=subject, trust_remote_code=True)
    except Exception as e:
        print(f"[C-Eval] Failed to load subject '{subject}': {e}")
        return []
    if "val" not in ds or len(ds["val"]) == 0:
        return []
    return list(ds["val"])


def _stratified_split(
    all_rows: List[Dict[str, Any]],
    subject_labels: List[str],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[Dict, str]], List[Tuple[Dict, str]]]:
    """
    Split (row, subject) pairs into train/val, stratified by subject.
    Returns lists of (row, subject) tuples.
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    by_subject: Dict[str, List[Tuple[Dict, str]]] = defaultdict(list)
    for row, subj in zip(all_rows, subject_labels):
        by_subject[subj].append((row, subj))

    val_pairs, train_pairs = [], []
    for pairs in by_subject.values():
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_pairs.extend(shuffled[:n_val])
        train_pairs.extend(shuffled[n_val:])

    return train_pairs, val_pairs


def load_ceval_examples(
    n: Optional[int] = None,
    split: str = "val",
    val_fraction: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Load C-Eval examples for Chinese.

    C-Eval only has a labeled 'val' split (~1346 examples across 52 subjects).
    We further split that into our own train/val pools.

    split='val'   → held-out validation pool
    split='train' → training pool
    """
    subjects = _get_all_subjects()

    all_rows, subject_labels = [], []
    for subj in subjects:
        rows = _load_subject_val(subj)
        all_rows.extend(rows)
        subject_labels.extend([subj] * len(rows))

    train_pairs, val_pairs = _stratified_split(
        all_rows, subject_labels, val_fraction, VAL_SEED
    )

    pairs = val_pairs if split == "val" else train_pairs

    # Respect n cap before formatting
    if n is not None:
        pairs = pairs[:n]

    return [
        _format_example(row, subj, idx)
        for idx, (row, subj) in enumerate(pairs)
    ]


def load_ceval_splits() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience: return (train_examples, val_examples)."""
    train = load_ceval_examples(n=None, split="train")
    val   = load_ceval_examples(n=None, split="val")
    return train, val