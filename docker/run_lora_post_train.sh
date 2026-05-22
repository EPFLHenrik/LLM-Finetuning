#!/usr/bin/env bash
set -euo pipefail

# Submit a Run:AI job to run LoRA post-training on the cluster.
# Mirrors the style of other scripts in this folder and supports a local
# quick-run mode with `--local`.

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_lora_post_train.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_lora_post_train.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

SUFFIX="lora-train"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

# Defaults -- feel free to override before submitting
IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Local quick run (no job submission). Use for small smoke-tests.
# Expose all script parameters as variables (mandatory ones left empty)
# Mandatory (edit before submitting)
MODEL_FOLDER="models/qwen3_thinking"         # REQUIRED: e.g. models/qwen3_baseline
DATASET_FILE="data/preprocessing/commonsense_qa/train.jsonl"         # REQUIRED: path to .jsonl or .json
OUTPUT_DIR="models/commonsense_qa"           # REQUIRED: where merged checkpoint will be written

# Optional / defaults (match lora_post_train.py defaults)
RUN_NAME="first_commonsense"
SEED=42
MAX_SEQ_LENGTH=2048
NUM_TRAIN_EPOCHS=1.0
LEARNING_RATE=0.0002
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=10
SAVE_STRATEGY="steps"
SAVE_STEPS=500
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
MAX_TRAIN_SAMPLES=0
WANDB_PROJECT="cs552-mystery-machine"
SKIP_WANDB=false
OVERWRITE_OUTPUT=true
MERGE_AND_SAVE=true

# Build flag args
SKIP_WANDB_ARG=""
OVERWRITE_OUTPUT_ARG="--overwrite-output"
MERGE_AND_SAVE_ARG="--merge-and-save"
WANDB_KEY="${WANDB_KEY:-}"
 

if [ "$SKIP_WANDB" = "true" ]; then SKIP_WANDB_ARG="--skip-wandb"; fi
if [ "$OVERWRITE_OUTPUT" = "true" ]; then OVERWRITE_OUTPUT_ARG="--overwrite-output"; fi
if [ "$MERGE_AND_SAVE" = "false" ]; then MERGE_AND_SAVE_ARG=""; fi

# Check mandatory variables
if [ -z "$MODEL_FOLDER" ] || [ -z "$DATASET_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Please set MODEL_FOLDER, DATASET_FILE, and OUTPUT_DIR near the top of this script before submitting."
  exit 1
fi

echo ">>> Submitting ${JOB_NAME} (LoRA post-training job)"

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
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"${REPO_DIR:-/scratch/workspace/henrik/standard-project-m2-mystery-machine}\" && \
    python training/lora_post_train.py \
      --model-folder \"${MODEL_FOLDER}\" \
      --dataset-file \"${DATASET_FILE}\" \
      --output-dir \"${OUTPUT_DIR}\" \
      --run-name "${RUN_NAME}" \
      --seed ${SEED} \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --num-train-epochs ${NUM_TRAIN_EPOCHS} \
      --learning-rate ${LEARNING_RATE} \
      --per-device-train-batch-size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
      --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
      --logging-steps ${LOGGING_STEPS} \
      --save-strategy "${SAVE_STRATEGY}" \
      --save-steps ${SAVE_STEPS} \
      --lora-r ${LORA_R} \
      --lora-alpha ${LORA_ALPHA} \
      --lora-dropout ${LORA_DROPOUT} \
      --max-train-samples ${MAX_TRAIN_SAMPLES} \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-key "${WANDB_KEY}" \
      ${SKIP_WANDB_ARG} \
      ${OVERWRITE_OUTPUT_ARG} \
      ${MERGE_AND_SAVE_ARG}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF
