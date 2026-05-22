"""multilingual/loaders/recarve_utils.py
Shared utility for datasets that need val/test recarving from a single HF pool.
Used by include_loader and global_piqa_loader.
"""

from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset, concatenate_datasets
from multilingual.utils import VAL_SEED

DEFAULT_VAL_FRACTION = 0.5  # 50/50 default for recarve datasets


def recarve_splits(
    hf_datasets: List[Dataset],
    n_val: Optional[int],
    n_test: Optional[int],
    dataset_name: str,
    language: str,
) -> Tuple[Dataset, Dataset] | Tuple[None, None]:
    """
    Merge HF datasets, shuffle, and recarve into (val_data, test_data).

    Args:
        hf_datasets: List of HF Dataset objects to merge (e.g. [ds["test"]]).
        n_val:       Max val examples. None → DEFAULT_VAL_FRACTION of merged.
        n_test:      Max test examples. None → remainder after val.
        dataset_name: For logging only.
        language:    For logging only.

    Returns:
        (val_data, test_data) as HF Dataset objects.
    """
    if not hf_datasets:
        print(f"[{dataset_name}] No data to recarve for {language}.")
        return None, None

    merged  = concatenate_datasets(hf_datasets).shuffle(seed=VAL_SEED)
    n_total = len(merged)

    if n_val is None:
        n_val = max(1, int(n_total * DEFAULT_VAL_FRACTION))
    n_val = min(n_val, n_total)

    if n_test is None:
        n_test = n_total - n_val
    n_test = min(n_test, n_total - n_val)

    val_data  = merged.select(range(n_val))
    test_data = merged.select(range(n_val, n_val + n_test))
    return val_data, test_data