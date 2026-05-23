#!/usr/bin/env bash
set -euo pipefail

# Submit a Run:AI job that resamples incorrect answers with the local model.
# Mirrors the style of the other scripts in this folder.

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_resample_wrong_answers.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_resample_wrong_answers.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

SUFFIX="resample-wrong-answers"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

# Default image with CUDA, PyTorch, and common libraries.
IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Inputs / outputs for the resampling job.
MODEL_DIR="/scratch/workspace/henrik/standard-project-m2-mystery-machine/models/qwen3_thinking"
INPUT_FILE="/scratch/workspace/henrik/standard-project-m2-mystery-machine/data/preprocessing/commonsense_qa/train_student.jsonl"
OUTPUT_FILE="/scratch/workspace/henrik/standard-project-m2-mystery-machine/data/preprocessing/commonsense_qa/student_corrected.jsonl"
MAX_NEW_TOKENS=4096
USE_STUDENT_FLAG="true"

if [[ -z "${MODEL_DIR}" || -z "${INPUT_FILE}" || -z "${OUTPUT_FILE}" ]]; then
  echo "Please set MODEL_DIR, INPUT_FILE, and OUTPUT_FILE near the top of this script before submitting."
  exit 1
fi

STUDENT_ARG=""
if [[ "${USE_STUDENT_FLAG}" == "true" ]]; then
  STUDENT_ARG="--student"
fi

echo ">>> Submitting ${JOB_NAME} (resample wrong answers job)"

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
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"\${REPO_DIR:-/scratch/workspace/henrik/standard-project-m2-mystery-machine}\" && \
    python training/resample_wrong_answers.py \
      \"${INPUT_FILE}\" \
      --model-dir \"${MODEL_DIR}\" \
      --output \"${OUTPUT_FILE}\" \
      --hf-token \"${HF_TOKEN:-}\" \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      ${STUDENT_ARG}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF