#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root
DATA_DIR="$PROJECT_DIR/data/general_knowledge"
PREPROCESSING_DIR="$DATA_DIR/processed"

python "$GK_DIR/scripts/build_dpo_dataset.py" \
    --runs \
        "$PREPROCESSING_DIR/grpo_runs/run1.jsonl" \
        "$PREPROCESSING_DIR/grpo_runs/run2.jsonl" \
        "$PREPROCESSING_DIR/grpo_runs/run3.jsonl" \
    --distillation "$PREPROCESSING_DIR/merged_distillations.jsonl" \
    --output "$PREPROCESSING_DIR/dpo_pairs.jsonl"
