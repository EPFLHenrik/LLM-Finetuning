#!/bin/bash
set -euo pipefail
# Submit a Run:AI job to run scripts/model_config.py with fixed parameters.
# Styled after docker/run_setup_baseline.sh: edit constants in this file, then run.
#
# Usage (local):
#   ./docker/run_model_config.sh --local
#
# Usage (submit job):
#   ./docker/run_model_config.sh

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"                    # <-- YOUR GASPAR EPFL username.
GROUP="g02"                         # <-- YOUR TEAM, e.g. g07.

# model_config.py arguments as constants
MODEL_PATH="models/qwen3_baseline2"  # Source checkpoint dir (relative to WORKSPACE_BASE)
OUTPUT_DIR="models/qwen3_thinking"   # Output checkpoint dir (relative to WORKSPACE_BASE)
CMD="chat_template"                  # all | config | generation_config | tokenizer_config | chat_template
TRUST_REMOTE_CODE="true"            # true | false
USE_FAST="true"                     # true | false
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_model_config.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_model_config.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

WORKSPACE_BASE="/scratch/workspace/henrik/standard-project-m2-mystery-machine"
REPO_DIR="${WORKSPACE_BASE}"
GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="model-config"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

MODEL_PATH="${MODEL_PATH#/}"
OUTPUT_DIR="${OUTPUT_DIR#/}"

if [[ "$MODEL_PATH" == *".."* || "$OUTPUT_DIR" == *".."* ]]; then
  echo "Refusing MODEL_PATH/OUTPUT_DIR containing '..'" >&2
  exit 1
fi

MODEL_PATH_ABS="${WORKSPACE_BASE}/${MODEL_PATH}"
OUTPUT_DIR_ABS="${WORKSPACE_BASE}/${OUTPUT_DIR}"

EXTRA_FLAGS=()
if [[ "${TRUST_REMOTE_CODE}" != "true" ]]; then
  EXTRA_FLAGS+=("--no-trust-remote-code")
fi
if [[ "${USE_FAST}" != "true" ]]; then
  EXTRA_FLAGS+=("--no-fast")
fi

# Allow quick local run without submitting a job
if [[ "${1:-}" == "--local" ]]; then
  shift || true
  exec python scripts/model_config.py \
    --model-path "${MODEL_PATH_ABS}" \
    --output-dir "${OUTPUT_DIR_ABS}" \
    --cmd "${CMD}" \
    "${EXTRA_FLAGS[@]}"
fi

echo ">>> Submitting ${JOB_NAME} to adapt checkpoint"
echo ">>> Source: ${MODEL_PATH_ABS}"
echo ">>> Output: ${OUTPUT_DIR_ABS}"

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
  python3 scripts/model_config.py \
  --model-path "${MODEL_PATH_ABS}" \
  --output-dir "${OUTPUT_DIR_ABS}" \
  --cmd "${CMD}" \
  "${EXTRA_FLAGS[@]}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF