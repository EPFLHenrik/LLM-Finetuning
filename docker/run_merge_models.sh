#!/usr/bin/env bash
set -euo pipefail

# Submit a Run:AI job to merge two model checkpoints (LERP or SLERP).
# Runs on CPU so no GPU is needed — uses a standard CPU node.

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_merge_models.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_merge_models.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

SUFFIX="merge-models"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# ---- Edit these ----
MODEL_A="models/full_sft"           # REQUIRED: first model (tokenizer copied from here)
MODEL_B="models/full_math_grpo"     # REQUIRED: second model
OUTPUT_DIR="models/merged_sft_grpo" # REQUIRED: output path

ALPHA=0.5                           # 0 = all model-a, 1 = all model-b
METHOD="lerp"                       # lerp or slerp
DEVICE="cpu"                        # cpu keeps GPU free; use cuda to speed up on large models

# ---- Validate ----
if [ -z "$MODEL_A" ] || [ -z "$MODEL_B" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: set MODEL_A, MODEL_B, and OUTPUT_DIR before submitting." >&2
    exit 1
fi

echo ">>> Submitting ${JOB_NAME} (model merge: ${METHOD} alpha=${ALPHA})"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --cpu 8 \
  --memory 64G \
  --large-shm \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"${REPO_DIR:-/scratch/workspace/henrik/standard-project-m2-mystery-machine}\" && \
    python scripts/merge_models.py \
      --model-a \"${MODEL_A}\" \
      --model-b \"${MODEL_B}\" \
      --output  \"${OUTPUT_DIR}\" \
      --alpha   ${ALPHA} \
      --method  ${METHOD} \
      --device  ${DEVICE}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}
EOF
