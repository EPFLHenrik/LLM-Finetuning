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
ALLOWED_FILES = (
    "chat_template.jinja",
    "checkpoint_manifest.json",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


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
    upload_paths = []
    for filename in ALLOWED_FILES:
        file_path = model_dir / filename
        if not file_path.is_file():
            print(f"[warn] Skipping missing file: {file_path}")
            continue
        upload_paths.append(file_path)

    if not upload_paths:
        raise FileNotFoundError(f"No allowed files found in {model_dir}")

    print(f"Uploading {len(upload_paths)} files from {model_dir} to {TARGET_REPO_ID}...")
    for file_path in upload_paths:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=TARGET_REPO_ID,
            repo_type="model",
            token=token,
        )
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))