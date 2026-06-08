#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "${REPO_DIR}"
source code/multilinguality_model/setup_env.sh

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set" >&2
  exit 1
fi
echo "Syncing SFT data and benchmarks"

python - <<'PY'
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download
from multilinguality_model.paths import get_repo_root

HF_DATASET_REPO = "cs-552-2026-mystery-machine/multilingual"
repo = get_repo_root()

def copy_tree(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory in HF snapshot: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source_path in source_dir.rglob("*"):
        if not source_path.is_file():
            continue
        rel = source_path.relative_to(source_dir)
        target_path = target_dir / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    print(f"{target_dir}: copied {copied} files")

versions = ["v1", "v20-russian", "v20-hindi"]
patterns = [f"sft_data/{version}/**" for version in versions] + ["benchmarks/**"]

snapshot = Path(
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=patterns,
    )
)

for version in versions:
    copy_tree(
        snapshot / "sft_data" / version,
        repo / "data" / "multilinguality" / "sft" / version,
    )

copy_tree(
    snapshot / "benchmarks",
    repo / "data" / "multilinguality" / "benchmarks",
)
PY

for version in v1 v20-russian v20-hindi; do
  test -f "data/multilinguality/sft/${version}/train_mixed.jsonl"
  test -f "data/multilinguality/sft/${version}/val_mixed.jsonl"
done

test -f "data/multilinguality/benchmarks/benchmark_all.jsonl"

echo "Data sync complete"
