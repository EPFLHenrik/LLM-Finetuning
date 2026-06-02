#!/bin/bash
# Submit one RCP job:
#   1. Train individual v16 non-thinking model
#   2. Export and push to HF
#
# v16 uses:
#   - non-thinking boxed-answer-only SFT data
#   - --system_prompt non_thinking
#   - no group model training

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="anastass"
GROUP="g02"
REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"

SFT_CONFIG="${REPO_DIR}/configs/sft/v16.json"
SFT_DATA_DIR="${REPO_DIR}/multilingual/data/sft/v16"
TRAIN_FILE="${SFT_DATA_DIR}/mixed/train_mixed.jsonl"
VAL_FILE="${SFT_DATA_DIR}/mixed/val_mixed.jsonl"

HF_REPO="cs-552-2026-mystery-machine/multilingual_model"
RUN_NAME="sft_multilingual_v16_nonthinking_1e-4_warmup0p1"

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
EPOCHS=3
LR="1e-4"
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

if [[ ! -f "${TRAIN_FILE}" ]]; then echo "ERROR: train file not found: ${TRAIN_FILE}" >&2; exit 1; fi
if [[ ! -f "${VAL_FILE}" ]]; then echo "ERROR: val file not found: ${VAL_FILE}" >&2; exit 1; fi
if [[ ! -f "${SFT_CONFIG}" ]]; then echo "ERROR: config file not found: ${SFT_CONFIG}" >&2; exit 1; fi

GPUS=1
NODE="${NODE:-a100-40g}"
PROJECT="course-cs-552-${GASPAR}"
WANDB_ENTITY="nutemmax-epfl"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

OUTPUT_DIR="${REPO_DIR}/multilingual/results/checkpoints/${RUN_NAME}/individual"
ADAPTER_DIR="${OUTPUT_DIR}/best_adapter"
EXPORT_DIR="${REPO_DIR}/multilingual/results/exports/${RUN_NAME}"

# ---------------------------------------------------------------------------
# Single job: train individual non-thinking model, then export
# ---------------------------------------------------------------------------
SUFFIX="${1:-sft-v16-nonthinking-export}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"

echo ">>> Submitting job: ${JOB_NAME}"
echo ">>> Training individual v16 non-thinking model → export → HF push"
echo ">>> Train file: ${TRAIN_FILE}"
echo ">>> Val file:   ${VAL_FILE}"
echo ">>> Output dir: ${OUTPUT_DIR}"

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
    echo '=== Step 1: Training individual v16 non-thinking model ===' && \
    python -m multilingual.train.sft \
      --run_name ${RUN_NAME} \
      --output_dir ${OUTPUT_DIR} \
      --train_file ${TRAIN_FILE} \
      --val_file ${VAL_FILE} \
      --wandb_entity ${WANDB_ENTITY} \
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
      --per_device_eval_batch_size ${EVAL_BATCH_SIZE} && \
    echo '=== Step 2: Exporting and pushing v16 non-thinking model to HF ===' && \
    python -m multilingual.train.export \
      --adapter_path ${ADAPTER_DIR} \
      --output_dir ${EXPORT_DIR} \
      --run_name ${RUN_NAME} \
      --hf_repo ${HF_REPO} \
      --override_chat_template ${REPO_DIR}/multilingual/assets/chat_template_multilingual-v2.jinja \
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