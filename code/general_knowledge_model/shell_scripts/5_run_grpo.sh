#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root
DATA_DIR="$PROJECT_DIR/data"
TRAINING_DIR="$GK_DIR/training"
PREPROCESSING_DIR="$DATA_DIR/preprocessing"

python "$TRAINING_DIR/grpo_train.py" \
  --model-folder "$PROJECT_DIR/models/full_sft" \
  --dataset-file "$PREPROCESSING_DIR/grpo_runs/partial_success.jsonl" \
  --output-dir "$PROJECT_DIR/models/sft_grpo" \
  --use-lora \
  --overwrite-output \
  --save-strategy steps \
  --save-steps 10000 \
  --learning-rate 1e-6 \
  --beta 0.08 \
  --format-weight 0.1 \
  --acc-weight 0.9
