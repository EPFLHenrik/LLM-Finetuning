"""code/multilinguality_model/data_pipeline/download_processed_data.py
Sync processed data from the HF dataset repo.
"""

from __future__ import annotations
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download
from multilinguality_model.paths import get_processed_data_dir


HF_DATASET_REPO = "cs-552-2026-mystery-machine/multilingual"
HF_PROCESSED_SUBDIR = "processed_data"
_SYNCED = False
_SYNC_MARKER = ".hf_processed_data_synced"


def _has_local_processed_data(processed_dir: Path) -> bool:
    if not processed_dir.exists():
        return False
    marker = processed_dir / _SYNC_MARKER
    if marker.exists():
        return True
    return any(processed_dir.glob("*/*.jsonl"))


def ensure_processed_data_available(force: bool = False) -> Path:
    global _SYNCED
    processed_dir = get_processed_data_dir()
    if _SYNCED and not force:
        return processed_dir
    if _has_local_processed_data(processed_dir) and not force:
        _SYNCED = True
        return processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=f"{HF_PROCESSED_SUBDIR}/**",
        )
    )

    source_dir = snapshot_path / HF_PROCESSED_SUBDIR
    if not source_dir.exists():
        available = sorted(p.name for p in snapshot_path.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"Could not find {HF_PROCESSED_SUBDIR!r} in HF snapshot {snapshot_path}. "
            f"Available folders: {available}"
        )

    copied = 0
    for source_path in source_dir.rglob("*"):
        if not source_path.is_file():
            continue

        relative_path = source_path.relative_to(source_dir)
        target_path = processed_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
            continue

        shutil.copy2(source_path, target_path)
        copied += 1

    (processed_dir / _SYNC_MARKER).write_text(
        f"source=hf://datasets/{HF_DATASET_REPO}/{HF_PROCESSED_SUBDIR}\n",
        encoding="utf-8",
    )

    if copied:
        print(f"Synced {copied} processed files from HF to {processed_dir}")
    _SYNCED = True
    return processed_dir
