#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root
DATA_DIR="$PROJECT_DIR/data/general_knowledge"
TRAINING_DIR="$GK_DIR/training"
PREPROCESSING_DIR="$DATA_DIR/processed"

python "$TRAINING_DIR/dpo_train.py" \
    --model-folder "$PROJECT_DIR/checkpoints/general_knowledge_model/sft_grpo" \
    --dataset-file "$PREPROCESSING_DIR/dpo_pairs.jsonl" \
    --output-dir "$PROJECT_DIR/checkpoints/general_knowledge_model/sft_grpo_dpo" \
    --beta 0.1 \
    --learning-rate 1e-6 \
    --num-train-epochs 1.0 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --use-lora \
    --overwrite-output \
    --wandb-key "${WANDB_API_KEY:?WANDB_API_KEY environment variable is required}"
