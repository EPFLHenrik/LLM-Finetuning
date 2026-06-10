#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root
DATA_DIR="$PROJECT_DIR/data/general_knowledge"
TRAINING_DIR="$GK_DIR/training"
PREPROCESSING_DIR="$DATA_DIR/processed"
BASE_MODEL="$PROJECT_DIR/checkpoints/general_knowledge_model/qwen_thinking3"

python "$GK_DIR/scripts/model_config.py" \
    --model-path "$BASE_MODEL" \
    --cmd chat_template \
    --in-place

python "$TRAINING_DIR/lora_post_train.py" \
    --dataset-file "$PREPROCESSING_DIR/train_datasets/merged_distillations.jsonl" \
    --model-folder "$PROJECT_DIR/checkpoints/general_knowledge_model/qwen_thinking3" \
    --output-dir "$PROJECT_DIR/checkpoints/general_knowledge_model/full_sft" \
    --run-name sft_full_math \
    --wandb-key "${WANDB_API_KEY:?WANDB_API_KEY environment variable is required}" \
    --merge-and-save \
    --overwrite-output
