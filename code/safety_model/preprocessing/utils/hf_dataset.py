from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk


def load_shuffled_dataset(path: Path, seed: int):
    dataset = load_from_disk(str(path))
    return dataset.shuffle(seed=seed)


def compute_effective_range(
    from_id: int | None,
    to_id: int | None,
    dataset_len: int,
) -> tuple[int, int]:
    effective_from = 0 if from_id is None else from_id
    requested_to = dataset_len if to_id is None else to_id

    if effective_from < 0 or requested_to < 0:
        raise ValueError("--from_id and --to_id must be >= 0")

    if effective_from >= requested_to:
        raise ValueError("--from_id must be strictly smaller than --to_id")

    effective_to = min(requested_to, dataset_len)
    return effective_from, effective_to


def build_pending_indices(
    effective_from: int,
    effective_to: int,
    processed_ids: set[int],
    overwrite: bool,
) -> list[int]:
    if overwrite:
        return list(range(effective_from, effective_to))

    return [
        idx
        for idx in range(effective_from, effective_to)
        if idx not in processed_ids
    ]