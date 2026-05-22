#!/usr/bin/env bash
#!/bin/bash
set -euo pipefail

# Submit a Run:AI job to evaluate commonsense_qa LoRA adapter checkpoints.
# Mirrors the style of other scripts in this folder and supports a local
# quick-run mode with `--local`.

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_evaluate_commonsense_qa_lora_adapter.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_evaluate_commonsense_qa_lora_adapter.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

SUFFIX="eval-lora-adapter"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

# Defaults -- feel free to override before submitting
IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Default paths inside the repo
CHECKPOINTS_DIR="models/commonsense_qa/lora_adapter"
OUTPUT_DIR="results/commonsense_qa/lora_adapter"
VALIDATION_FILE="data/preprocessing/commonsense_qa/validation.jsonl"
DEFAULT_BASE_MODEL="models/commonsense_qa/lora_adapter/base_model"

# Local quick run: call the small runner script (uses DEFAULT_BASE_MODEL)
if [[ "${1:-}" == "--local" ]]; then
  shift || true
  exec bash scripts/run_evaluate_commonsense_qa_lora_adapter.sh "$@"
fi

# Optional first arg overrides base model path
BASE_MODEL="${1:-$DEFAULT_BASE_MODEL}"

echo ">>> Submitting ${JOB_NAME} (evaluation job)"

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
    python scripts/evaluate_training.py \
      --checkpoints-dir \"${CHECKPOINTS_DIR}\" \
      --base-model \"${BASE_MODEL}\" \
      --validation-file \"${VALIDATION_FILE}\" \
      --output-dir \"${OUTPUT_DIR}\" \
      --max-new-tokens 512 \
      --temperature 0.0 \
      --top-p 0.95 \
      --wandb-key "${WANDB_KEY}" \
      --max-eval-samples 0"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF
