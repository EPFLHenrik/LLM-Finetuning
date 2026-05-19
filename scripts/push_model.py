#!/usr/bin/env python3
"""Push a local model directory to Hugging Face.

Usage:
    python scripts/push_model.py /path/to/model_dir

Target repository is fixed to:
    cs-552-2026-AttentionSeekers/general_knowledge_model
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


TARGET_REPO_ID = "cs-552-2026-mystery-machine/general_knowledge_model"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Push a model directory to Hugging Face")
    parser.add_argument("model_dir", type=Path, help="Local model directory to upload")
    args = parser.parse_args(argv)

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")

    api = HfApi()
    api.create_repo(repo_id=TARGET_REPO_ID, repo_type="model", exist_ok=True, token=token)
    print(f"Uploading {model_dir} to {TARGET_REPO_ID}...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=TARGET_REPO_ID,
        repo_type="model",
        commit_message=f"Upload {model_dir.name}",
        token=token,
    )
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))