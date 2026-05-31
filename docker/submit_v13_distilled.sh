#!/bin/bash
# Submit two sequential RCP jobs:
#   1. Individual model: trained with v2 system prompt, exported and pushed to HF
#   2. Group model: trained with group system prompt, saved locally only (no HF push)
#
# Checkpoint layout:
#   checkpoints/<RUN_NAME>/individual/  — individual model checkpoints + best_adapter
#   checkpoints/<RUN_NAME>/group/       — group model checkpoints + best_adapter

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="anastass"
GROUP="g02"
REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"

SFT_CONFIG="${REPO_DIR}/configs/sft/v13.json"
SFT_DATA_DIR="${REPO_DIR}/multilingual/data/sft/v13"
TRAIN_FILE="${SFT_DATA_DIR}/mixed/train_mixed.jsonl"
VAL_FILE="${SFT_DATA_DIR}/mixed/val_mixed.jsonl"

HF_REPO="cs-552-2026-mystery-machine/multilingual_model"
RUN_NAME="sft_multilingual_v13_distilled"

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
EPOCHS=3
LR="3e-5"
BATCH_SIZE=2
GRAD_ACCUM=8
EVAL_STEPS=100
SAVE_STEPS=100
EVAL_BATCH_SIZE=2
LOG_STEPS=10
WARMUP_RATIO=0.1
# ==============================================

if [[ -z "${GASPAR}" ]]; then echo "ERROR: set GASPAR." >&2; exit 1; fi
if [[ -z "${GROUP}" ]]; then echo "ERROR: set GROUP." >&2; exit 1; fi
if [[ -z "${HF_TOKEN:-}" ]]; then echo "ERROR: HF_TOKEN not set." >&2; exit 1; fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then echo "ERROR: WANDB_API_KEY not set." >&2; exit 1; fi

GPUS=1
NODE="${NODE:-a100-40g}"
PROJECT="course-cs-552-${GASPAR}"
WANDB_ENTITY="nutemmax-epfl"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

INDIVIDUAL_OUTPUT_DIR="${REPO_DIR}/multilingual/results/checkpoints/${RUN_NAME}/individual"
GROUP_OUTPUT_DIR="${REPO_DIR}/multilingual/results/checkpoints/${RUN_NAME}/group"
INDIVIDUAL_ADAPTER="${INDIVIDUAL_OUTPUT_DIR}/best_adapter"
EXPORT_DIR="${REPO_DIR}/multilingual/results/exports/${RUN_NAME}"

# ---------------------------------------------------------------------------
# Single job: individual first, then group sequentially
# ---------------------------------------------------------------------------
JOB_NAME="cs552-${GASPAR}-${GROUP}-${RUN_NAME}-$(date +%H%M%S)"

echo ">>> Submitting job: ${JOB_NAME}"
echo ">>> Individual → export → Group (sequential)"

MSYS_NO_PATHCONV=1 runai training submit "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu-devices-request "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN}" \
  --environment WANDB_API_KEY="${WANDB_API_KEY}" \
  --environment WANDB_DIR=/scratch/wandb \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd ${REPO_DIR} && \
    echo '=== Step 1: Training individual model ===' && \
    python -m multilingual.train.sft \
      --run_name ${RUN_NAME}_individual \
      --output_dir ${INDIVIDUAL_OUTPUT_DIR} \
      --train_file ${TRAIN_FILE} \
      --val_file ${VAL_FILE} \
      --wandb_entity ${WANDB_ENTITY} \
      --system_prompt individual \
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
      --per_device_eval_batch_size ${EVAL_BATCH_SIZE} && \
    echo '=== Step 2: Exporting and pushing individual model to HF ===' && \
    python -m multilingual.train.export \
      --adapter_path ${INDIVIDUAL_ADAPTER} \
      --output_dir ${EXPORT_DIR} \
      --run_name ${RUN_NAME} \
      --hf_repo ${HF_REPO} \
      --push \
      --force_push && \
    echo '=== Step 3: Training group model ===' && \
    python -m multilingual.train.sft \
      --run_name ${RUN_NAME}_group \
      --output_dir ${GROUP_OUTPUT_DIR} \
      --train_file ${TRAIN_FILE} \
      --val_file ${VAL_FILE} \
      --wandb_entity ${WANDB_ENTITY} \
      --system_prompt group \
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
      --per_device_eval_batch_size ${EVAL_BATCH_SIZE}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch logs:
  runai workload logs ${JOB_NAME} -p ${PROJECT} -f

List workloads:
  runai workload list -p ${PROJECT}

EOF