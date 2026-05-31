#!/bin/bash
# Submit one RCP job that builds v12 SFT data, runs 1-epoch multilingual SFT,
# exports the best LoRA adapter, merges it into Qwen/Qwen3-1.7B,
# and force-pushes the merged checkpoint to Hugging Face.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="anastass"
GROUP="g02"
REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"

SFT_CONFIG="${REPO_DIR}/configs/sft/v12.json"
SFT_DATA_DIR="${REPO_DIR}/multilingual/data/sft/v12"
TRAIN_FILE="${SFT_DATA_DIR}/mixed/train_mixed.jsonl"
VAL_FILE="${SFT_DATA_DIR}/mixed/val_mixed.jsonl"

HF_REPO="cs-552-2026-mystery-machine/multilingual_model"
RUN_NAME="sft_multilingual_v12_distilled_only_short" # for the shorter run, i used 2 epochs

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
EPOCHS=1
LR="3e-5"
BATCH_SIZE=2
GRAD_ACCUM=8
EVAL_STEPS=100
SAVE_STEPS=100
EVAL_BATCH_SIZE=2
LOG_STEPS=10
WARMUP_RATIO=0.05
# ==============================================

if [[ -z "${GASPAR}" ]]; then
    echo "ERROR: set GASPAR." >&2
    exit 1
fi

if [[ -z "${GROUP}" ]]; then
    echo "ERROR: set GROUP." >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set in your shell. Run: export HF_TOKEN=..." >&2
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set in your shell. Run: export WANDB_API_KEY=..." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-sft-v12-short-export}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"
WANDB_ENTITY="nutemmax-epfl"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

ADAPTER_PATH="${REPO_DIR}/multilingual/results/checkpoints/${RUN_NAME}/best_adapter"
EXPORT_DIR="${REPO_DIR}/multilingual/results/exports/${RUN_NAME}"

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Run name:   ${RUN_NAME}"
echo ">>> HF repo:    ${HF_REPO}"
echo ">>> SFT config: ${SFT_CONFIG}"
echo ">>> Train file: ${TRAIN_FILE}"
echo ">>> Val file:   ${VAL_FILE}"
echo ">>> Epochs:     ${EPOCHS}"
echo ">>> LR:         ${LR}"
echo ">>> Dropout:    ${LORA_DROPOUT}"

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
    python -m multilingual.train.sft \
      --run_name ${RUN_NAME} \
      --train_file ${TRAIN_FILE} \
      --val_file ${VAL_FILE} \
      --wandb_entity ${WANDB_ENTITY} \
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
    python -m multilingual.train.export \
      --adapter_path ${ADAPTER_PATH} \
      --output_dir ${EXPORT_DIR} \
      --run_name ${RUN_NAME} \
      --hf_repo ${HF_REPO} \
      --push \
      --force_push"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:
  runai workload describe ${JOB_NAME} -p ${PROJECT}

List workloads:
  runai workload list -p ${PROJECT}

Stream logs:
  runai workload logs ${JOB_NAME} -p ${PROJECT}

If streaming logs with the new command is not supported, use:
  runai logs -f ${JOB_NAME} -p ${PROJECT}

Shell in pod:
  runai bash ${JOB_NAME} -p ${PROJECT}

Attach with VS Code:
  refresh the Kubernetes sidebar and attach to pod ${JOB_NAME}-0-0

Stop the job:
  runai workload delete ${JOB_NAME} -p ${PROJECT}

EOF