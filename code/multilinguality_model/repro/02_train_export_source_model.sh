#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "${REPO_DIR}"
source code/multilinguality_model/setup_env.sh

MODEL_KEY="${1:?Usage: $0 {v1|v20-russian|v20-hindi}}"

BASE_MODEL="Qwen/Qwen3-1.7B"
CHAT_TEMPLATE="${REPO_DIR}/code/multilinguality_model/assets/chat_template_multilingual-nonthinking.jinja"

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
MAX_SEQ_LEN=4096

case "${MODEL_KEY}" in
  v1)
    SFT_VERSION="v1"
    RUN_NAME="v1"
    EPOCHS=1
    LR="2e-4"
    BATCH_SIZE=1
    GRAD_ACCUM=16
    EVAL_BATCH_SIZE=1
    EVAL_STEPS=250
    SAVE_STEPS=250
    LOG_STEPS=10
    WARMUP_RATIO="0.05"
    EXTRA_ARGS=(--preserve_system_prompt --use_qwen_default_chat_template)
    ;;
  v20-russian)
    SFT_VERSION="v20-russian"
    RUN_NAME="v20-russian"
    EPOCHS=2
    LR="2e-4"
    BATCH_SIZE=2
    GRAD_ACCUM=8
    EVAL_BATCH_SIZE=2
    EVAL_STEPS=100
    SAVE_STEPS=100
    LOG_STEPS=10
    WARMUP_RATIO="0.15"
    EXTRA_ARGS=(--system_prompt non_thinking)
    ;;
  v20-hindi)
    SFT_VERSION="v20-hindi"
    RUN_NAME="v20-hindi"
    EPOCHS=2
    LR="2e-4"
    BATCH_SIZE=2
    GRAD_ACCUM=8
    EVAL_BATCH_SIZE=2
    EVAL_STEPS=100
    SAVE_STEPS=100
    LOG_STEPS=10
    WARMUP_RATIO="0.15"
    EXTRA_ARGS=(--system_prompt non_thinking)
    ;;
  *)
    echo "Unknown model key: ${MODEL_KEY}" >&2
    exit 1
    ;;
esac

SFT_DIR="${REPO_DIR}/data/multilinguality/sft/${SFT_VERSION}"
TRAIN_FILE="${SFT_DIR}/train_mixed.jsonl"
VAL_FILE="${SFT_DIR}/val_mixed.jsonl"

OUTPUT_DIR="${REPO_DIR}/checkpoints/multilinguality_model/adapters/${RUN_NAME}/individual"
ADAPTER_DIR="${OUTPUT_DIR}/best_adapter"
EXPORT_DIR="${REPO_DIR}/checkpoints/multilinguality_model/${RUN_NAME}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set" >&2
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train file: ${TRAIN_FILE}" >&2
  exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing val file: ${VAL_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${EXPORT_DIR}" "${REPO_DIR}/logs"

echo "Training ${RUN_NAME}"
echo "Train: ${TRAIN_FILE}"
echo "Val:   ${VAL_FILE}"

python -m multilinguality_model.train.sft \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --wandb_entity "nutemmax-epfl" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --log_steps "${LOG_STEPS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"

echo "Exporting ${RUN_NAME}"

python -m multilinguality_model.train.export \
  --adapter_path "${ADAPTER_DIR}" \
  --output_dir "${EXPORT_DIR}" \
  --run_name "${RUN_NAME}" \
  --base_model "${BASE_MODEL}" \
  --override_chat_template "${CHAT_TEMPLATE}" \
  --overwrite \
  --skip_model_load

echo "Finished ${RUN_NAME}"
echo "Export: ${EXPORT_DIR}"
