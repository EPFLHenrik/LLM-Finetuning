#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root
DATA_DIR="$PROJECT_DIR/data"
TRAINING_DIR="$GK_DIR/training"
PREPROCESSING_DIR="$DATA_DIR/preprocessing"

python "$TRAINING_DIR/lora_post_train.py" \
    --dataset-file "$PREPROCESSING_DIR/merged_distillations.jsonl" \
    --model-folder "$PROJECT_DIR/models/qwen3_thinking" \
    --output-dir "$PROJECT_DIR/models/full_sft" \
    --run-name sft_full_math \
    --wandb-key "${WANDB_API_KEY:?WANDB_API_KEY environment variable is required}" \
    --merge-and-save \
    --overwrite-output
