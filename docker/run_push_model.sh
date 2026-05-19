#!/bin/bash
set -euo pipefail
# Submit a Run:AI job to run scripts/push_model.py with fixed parameters.
# Styled after docker/run_model_config.sh: edit constants in this file, then run.
#
# Usage (local):
#   ./docker/run_push_model.sh --local
#
# Usage (submit job):
#   ./docker/run_push_model.sh

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"                    # <-- YOUR GASPAR EPFL username.
GROUP="g02"                         # <-- YOUR TEAM, e.g. g07.

# push_model.py argument as constant
MODEL_DIR="models/qwen3_thinking"   # Model directory to upload (relative to WORKSPACE_BASE)
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_push_model.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_push_model.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

WORKSPACE_BASE="/scratch/workspace/henrik/standard-project-m2-mystery-machine"
REPO_DIR="${WORKSPACE_BASE}"
GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="push-model"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

MODEL_DIR="${MODEL_DIR#/}"
if [[ "$MODEL_DIR" == *".."* ]]; then
  echo "Refusing MODEL_DIR containing '..'" >&2
  exit 1
fi
MODEL_DIR_ABS="${WORKSPACE_BASE}/${MODEL_DIR}"

# Allow quick local run without submitting a job
if [[ "${1:-}" == "--local" ]]; then
  shift || true
  exec python scripts/push_model.py "${MODEL_DIR_ABS}"
fi

echo ">>> Submitting ${JOB_NAME} to push model"
echo ">>> Model dir: ${MODEL_DIR_ABS}"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir "${WORKSPACE_BASE}" \
  --environment HF_HOME="${WORKSPACE_BASE}/hf_cache" \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- \
  python3 scripts/push_model.py "${MODEL_DIR_ABS}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF
