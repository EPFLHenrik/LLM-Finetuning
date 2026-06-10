#!/usr/bin/env bash
set -euo pipefail

# Submit a Run:AI job to run DPO training on the cluster.
# Build the dataset first with scripts/build_dpo_dataset.py.

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_dpo_train.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_dpo_train.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

SUFFIX="dpo-train"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# ---- Mandatory (edit before submitting) ----
MODEL_FOLDER="models/full_sft"                               # REQUIRED: model to train from
DATASET_FILE="data/preprocessing/dpo_pairs.jsonl"            # REQUIRED: output of build_dpo_dataset.py
OUTPUT_DIR="models/full_sft_dpo"                             # REQUIRED: output path

# ---- Training ----
SEED=42
NUM_TRAIN_EPOCHS=1.0
LEARNING_RATE=1e-6
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
MAX_LENGTH=4096
MAX_PROMPT_LENGTH=1024
LOGGING_STEPS=20
SAVE_STRATEGY="epoch"
SAVE_STEPS=500
MAX_TRAIN_SAMPLES=0           # 0 = use all

# ---- DPO ----
BETA=0.1                      # KL penalty; higher = stay closer to reference model

# ---- LoRA ----
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# ---- Flags ----
GRADIENT_CHECKPOINTING=true
SKIP_WANDB=false
OVERWRITE_OUTPUT=true

# ---- W&B ----
WANDB_PROJECT="cs552-mystery-machine"
WANDB_ENTITY=""
WANDB_KEY="${WANDB_API_KEY:-}"

# ---- Build flag args ----
GRADIENT_CHECKPOINTING_ARG="--gradient-checkpointing"
SKIP_WANDB_ARG=""
OVERWRITE_OUTPUT_ARG=""
NO_LORA_ARG=""

if [ "$GRADIENT_CHECKPOINTING" = "false" ]; then GRADIENT_CHECKPOINTING_ARG="--no-gradient-checkpointing"; fi
if [ "$SKIP_WANDB" = "true" ];             then SKIP_WANDB_ARG="--skip-wandb"; fi
if [ "$OVERWRITE_OUTPUT" = "true" ];       then OVERWRITE_OUTPUT_ARG="--overwrite-output"; fi
if [ "$USE_LORA" = "false" ];             then NO_LORA_ARG="--no-lora"; fi

# ---- Validate ----
if [ -z "$MODEL_FOLDER" ] || [ -z "$DATASET_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: set MODEL_FOLDER, DATASET_FILE, and OUTPUT_DIR before submitting." >&2
    exit 1
fi

echo ">>> Submitting ${JOB_NAME} (DPO training)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu 1 \
  --large-shm \
  --node-pools "a100-40g" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment WANDB_DIR=/scratch/wandb \
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment PYTORCH_ALLOC_CONF=expandable_segments:True \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"${REPO_DIR:-/scratch/workspace/henrik/standard-project-m2-mystery-machine}\" && \
    python training/dpo_train.py \
      --model-folder \"${MODEL_FOLDER}\" \
      --dataset-file \"${DATASET_FILE}\" \
      --output-dir \"${OUTPUT_DIR}\" \
      --seed ${SEED} \
      --num-train-epochs ${NUM_TRAIN_EPOCHS} \
      --learning-rate ${LEARNING_RATE} \
      --per-device-train-batch-size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
      --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
      --max-length ${MAX_LENGTH} \
      --max-prompt-length ${MAX_PROMPT_LENGTH} \
      --logging-steps ${LOGGING_STEPS} \
      --save-strategy \"${SAVE_STRATEGY}\" \
      --save-steps ${SAVE_STEPS} \
      --beta ${BETA} \
      --max-train-samples ${MAX_TRAIN_SAMPLES} \
      --lora-r ${LORA_R} \
      --lora-alpha ${LORA_ALPHA} \
      --lora-dropout ${LORA_DROPOUT} \
      --lora-target-modules \"${LORA_TARGET_MODULES}\" \
      --wandb-project \"${WANDB_PROJECT}\" \
      --wandb-entity \"${WANDB_ENTITY}\" \
      --wandb-key \"${WANDB_KEY}\" \
      ${GRADIENT_CHECKPOINTING_ARG} \
      ${NO_LORA_ARG} \
      ${SKIP_WANDB_ARG} \
      ${OVERWRITE_OUTPUT_ARG}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF
