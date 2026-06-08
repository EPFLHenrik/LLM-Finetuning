"""code/multilinguality_model/data_pipeline/parsing/download_utils.py

Download raw parsing inputs from the HF dataset repo.
"""

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

from multilinguality_model.paths import get_raw_data_dir


HF_DATASET_REPO = "cs-552-2026-mystery-machine/multilingual"
HF_RAW_SUBDIR = "raw_data"


def ensure_raw_file(relative_path: str, download: bool = True) -> Path:
    local_path = get_raw_data_dir() / relative_path

    if local_path.exists():
        return local_path

    if not download:
        raise FileNotFoundError(
            f"Missing raw file: {local_path}\n"
            f"Pass without --no-download to fetch it from HF."
        )

    hf_filename = f"{HF_RAW_SUBDIR}/{relative_path}"

    cached_path = Path(
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=hf_filename,
        )
    )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, local_path)

    return local_path
