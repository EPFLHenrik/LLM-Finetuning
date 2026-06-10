#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root
DATA_DIR="$PROJECT_DIR/data/general_knowledge"
TRAINING_DIR="$GK_DIR/training"
PREPROCESSING_DIR="$DATA_DIR/processed"

HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable is required}"

python "$TRAINING_DIR/generate_answers.py" "$PREPROCESSING_DIR/full_train.jsonl" \
    --model-dir "$PROJECT_DIR/checkpoints/general_knowledge_model/full_sft" \
    --output "$PREPROCESSING_DIR/grpo_runs/run1.jsonl" \
    --hf-token "$HF_TOKEN" \
    --max-new-tokens 8192 \
    --seed 42 \
    --student \
    --max-model-len 10240 \
    --max-num-seqs 32

python "$TRAINING_DIR/generate_answers.py" "$PREPROCESSING_DIR/full_train.jsonl" \
    --model-dir "$PROJECT_DIR/checkpoints/general_knowledge_model/full_sft" \
    --output "$PREPROCESSING_DIR/grpo_runs/run2.jsonl" \
    --hf-token "$HF_TOKEN" \
    --max-new-tokens 8192 \
    --seed 123 \
    --student \
    --max-model-len 10240 \
    --max-num-seqs 32

python "$TRAINING_DIR/generate_answers.py" "$PREPROCESSING_DIR/full_train.jsonl" \
    --model-dir "$PROJECT_DIR/checkpoints/general_knowledge_model/full_sft" \
    --output "$PREPROCESSING_DIR/grpo_runs/run3.jsonl" \
    --hf-token "$HF_TOKEN" \
    --max-new-tokens 8192 \
    --seed 999 \
    --student \
    --max-model-len 10240 \
    --max-num-seqs 64

python "$GK_DIR/scripts/filter_partial_success.py" \
    "$PREPROCESSING_DIR/grpo_runs/run1.jsonl" \
    "$PREPROCESSING_DIR/grpo_runs/run2.jsonl" \
    "$PREPROCESSING_DIR/grpo_runs/run3.jsonl" \
    --output "$PREPROCESSING_DIR/grpo_runs/partial_success.jsonl"
