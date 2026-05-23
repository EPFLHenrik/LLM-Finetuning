#!/usr/bin/env python3
"""Download a Hugging Face teacher model snapshot into `models/teacher`.

This script mirrors the full repository layout from the Hub so the resulting
directory can be loaded directly by Transformers or vLLM.

Usage:
  python scripts/download_teacher_model.py
  python scripts/download_teacher_model.py --output-dir models/teacher
  python scripts/download_teacher_model.py --repo-id Qwen/Qwen3-14B-Instruct --yes
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_REPO_ID = "Qwen/Qwen3-14B-Instruct"
DEFAULT_OUTPUT_DIR = Path("models/teacher")


def _download_snapshot(repo_id: str, output_dir: Path, revision: str | None, token: str | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        token=token,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download a teacher model from Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repo id")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory")
    parser.add_argument("--revision", default=None, help="Optional repo revision, branch, tag, or commit")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--yes", action="store_true", help="Overwrite the output directory without prompting")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")

    if output_dir.exists():
        if not args.yes:
            response = input(f"Destination '{output_dir}' exists. Overwrite? [y/N] ")
            if response.strip().lower() not in {"y", "yes"}:
                print("Aborted")
                return 1
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} into {output_dir}...")
    snapshot_path = _download_snapshot(args.repo_id, output_dir, args.revision, token)
    print(f"Done: {snapshot_path}")
    print("The directory now contains the full model snapshot and can be loaded locally.")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))