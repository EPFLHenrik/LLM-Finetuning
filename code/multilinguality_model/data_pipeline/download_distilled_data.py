"""code/multilinguality_model/data_pipeline/download_distilled_data.py

Sync distilled data from the HF dataset repo.
"""

from __future__ import annotations
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download
from multilinguality_model.paths import get_distilled_data_dir

HF_DATASET_REPO = "cs-552-2026-mystery-machine/multilingual"
HF_DISTILLED_SUBDIR = "distilled_data"
_SYNCED = False
_SYNC_MARKER = ".hf_distilled_data_synced"


def _has_local_distilled_data(distilled_dir: Path) -> bool:
    if not distilled_dir.exists():
        return False
    if (distilled_dir / _SYNC_MARKER).exists():
        return True
    return any(distilled_dir.glob("**/*_distilled.jsonl"))


def ensure_distilled_data_available(force: bool = False) -> Path:
    global _SYNCED
    distilled_dir = get_distilled_data_dir()
    if _SYNCED and not force:
        return distilled_dir
    if _has_local_distilled_data(distilled_dir) and not force:
        _SYNCED = True
        return distilled_dir

    distilled_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=f"{HF_DISTILLED_SUBDIR}/**",
        )
    )
    source_dir = snapshot_path / HF_DISTILLED_SUBDIR
    if not source_dir.exists():
        available = sorted(p.name for p in snapshot_path.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"Could not find {HF_DISTILLED_SUBDIR!r} in HF snapshot {snapshot_path}. "
            f"Available folders: {available}"
        )

    copied = 0
    for source_path in source_dir.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source_dir)
        target_path = distilled_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
            continue
        shutil.copy2(source_path, target_path)
        copied += 1

    (distilled_dir / _SYNC_MARKER).write_text(
        f"source=hf://datasets/{HF_DATASET_REPO}/{HF_DISTILLED_SUBDIR}\n",
        encoding="utf-8",
    )
    if copied:
        print(f"Synced {copied} distilled files from HF to {distilled_dir}")
    _SYNCED = True
    return distilled_dir
