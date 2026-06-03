#!/bin/bash
# Train v17_20k non-thinking model, then export, push to HF, and evaluate.
# Run with: nohup bash train_v17_20k.sh > logs/train_v17_20k.log 2>&1 &

set -euo pipefail

REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"
SFT_DATA_DIR="${REPO_DIR}/multilingual/data/sft/v17_20k"
TRAIN_FILE="${SFT_DATA_DIR}/mixed/train_mixed.jsonl"
VAL_FILE="${SFT_DATA_DIR}/mixed/val_mixed.jsonl"

HF_REPO="cs-552-2026-mystery-machine/multilingual_model"
RUN_NAME="sft_multilingual_v17_20k_nonthinking_5e-5_warmup0p15"
CLEAN_NAME="${RUN_NAME#sft_multilingual_}"

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
EPOCHS=2
LR="5e-5"
BATCH_SIZE=2
GRAD_ACCUM=8
EVAL_STEPS=100
SAVE_STEPS=100
EVAL_BATCH_SIZE=2
LOG_STEPS=10
WARMUP_RATIO=0.2

OUTPUT_DIR="${REPO_DIR}/multilingual/results/checkpoints/${RUN_NAME}/individual"
ADAPTER_DIR="${OUTPUT_DIR}/best_adapter"
EXPORT_DIR="${REPO_DIR}/multilingual/results/exports/${RUN_NAME}"

CHAT_TEMPLATE="${REPO_DIR}/multilingual/assets/chat_template_multilingual-v2.jinja"
BENCHMARK_FILE="${REPO_DIR}/multilingual/data/benchmarks/benchmark_all.jsonl"
PROVIDED_FILE="${REPO_DIR}/validation_samples/multilingual.jsonl"

FULL_EVAL_DIR="${REPO_DIR}/multilingual/eval/results/${CLEAN_NAME}/full"
VALIDATION_EVAL_DIR="${REPO_DIR}/multilingual/eval/results/${CLEAN_NAME}/validation_set_mnlp"

mkdir -p "${OUTPUT_DIR}" "${EXPORT_DIR}" "${FULL_EVAL_DIR}" "${VALIDATION_EVAL_DIR}" "${REPO_DIR}/logs"

cd "${REPO_DIR}"

echo "========================================"
echo "Run:        ${RUN_NAME}"
echo "Clean name: ${CLEAN_NAME}"
echo "Train file: ${TRAIN_FILE}"
echo "Val file:   ${VAL_FILE}"
echo "Output:     ${OUTPUT_DIR}"
echo "Export:     ${EXPORT_DIR}"
echo "========================================"

echo "=== Step 1: Training v17_20k non-thinking model ==="
python -m multilingual.train.sft \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --train_file "${TRAIN_FILE}" \
    --val_file "${VAL_FILE}" \
    --wandb_entity "nutemmax-epfl" \
    --system_prompt non_thinking \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --log_steps ${LOG_STEPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE}

echo "=== Step 2: Exporting and pushing v17_20k to HF ==="
python -m multilingual.train.export \
    --adapter_path "${ADAPTER_DIR}" \
    --output_dir "${EXPORT_DIR}" \
    --run_name "${RUN_NAME}" \
    --hf_repo "${HF_REPO}" \
    --override_chat_template "${CHAT_TEMPLATE}" \
    --push \
    --force_push

echo "=== Step 3: Evaluating v17_20k on full benchmark ==="
python -m multilingual.eval.run_eval_vllm \
    --model_name "${EXPORT_DIR}" \
    --val_file "${BENCHMARK_FILE}" \
    --output_dir "${FULL_EVAL_DIR}" \
    --chat_template "${CHAT_TEMPLATE}" \
    --mcq_max_new_tokens 4096 \
    --math_max_new_tokens 4096 \
    --max_model_len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --overwrite \
    2>&1 | tee "${FULL_EVAL_DIR}/eval.log"

echo "=== Step 4: Evaluating v17_20k on MNLP validation samples ==="
python -m multilingual.eval.run_eval_vllm \
    --model_name "${EXPORT_DIR}" \
    --provided_only \
    --provided_file "${PROVIDED_FILE}" \
    --output_dir "${VALIDATION_EVAL_DIR}" \
    --chat_template "${CHAT_TEMPLATE}" \
    --mcq_max_new_tokens 4096 \
    --math_max_new_tokens 4096 \
    --max_model_len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --overwrite \
    2>&1 | tee "${VALIDATION_EVAL_DIR}/eval.log"

echo "=== Done: ${RUN_NAME} ==="
echo "Full eval summary:       ${FULL_EVAL_DIR}/summary.txt"
echo "Validation eval summary: ${VALIDATION_EVAL_DIR}/summary.txt"