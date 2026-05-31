from __future__ import annotations

import gc
import shutil
from pathlib import Path

from datasets import Dataset, load_from_disk


class DatasetCheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def exists(self) -> bool:
        return self.path.exists()

    def load_rows_by_id(self) -> dict[int, dict]:
        if not self.path.exists():
            return {}

        dataset_existing = load_from_disk(str(self.path))
        rows_existing = dataset_existing.to_list() # type: ignore
        del dataset_existing
        gc.collect()

        rows_by_id: dict[int, dict] = {}
        for pos, row in enumerate(rows_existing):
            raw_id = row.get("id")
            row_id = pos if raw_id is None else int(raw_id)
            row["id"] = row_id
            if row_id in rows_by_id:
                raise RuntimeError(
                    f"Duplicate row id={row_id} found in existing dataset at {self.path}."
                )
            rows_by_id[row_id] = row

        return rows_by_id

    def save_rows(self, rows_by_id: dict[int, dict]) -> None:
        rows_sorted = [rows_by_id[row_id] for row_id in sorted(rows_by_id)]
        self.save_rows_list(rows_sorted)

    def save_rows_list(self, rows: list[dict]) -> None:
        tmp_output = self.path.parent / f"{self.path.name}__tmp_save"
        if tmp_output.exists():
            shutil.rmtree(tmp_output)

        dataset_out = Dataset.from_list(rows)
        dataset_out.save_to_disk(str(tmp_output))

        if self.path.exists():
            shutil.rmtree(self.path)
        tmp_output.replace(self.path)