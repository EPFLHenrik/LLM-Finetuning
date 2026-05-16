#!/bin/bash
# CS-552 — submit a non-interactive Run:AI job to download and preprocess
# the CommonsenseQA dataset into /data.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                    # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit download.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit download.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-download}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

HF_TOKEN="${HF_TOKEN:-}"

REPO_DIR="/scratch/workspace/henrik/standard-project-m2-mystery-machine"

echo ">>> Submitting ${JOB_NAME} (1 GPU)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment WANDB_DIR=/scratch/wandb \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/data" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb /data/datasets/commonsense_qa /data/preprocessing/commonsense_qa && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"${REPO_DIR}\" && \
    python scripts/download_commonsense_qa.py --output-dir /data/datasets/commonsense_qa && \
    python scripts/preprocess_commonsense_qa.py --input-dir /data/datasets/commonsense_qa --output-dir /data/preprocessing/commonsense_qa"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

Output locations inside the pod:
  /data/datasets/commonsense_qa
  /data/preprocessing/commonsense_qa

The job runs both Python files in order:
  1. scripts/download_commonsense_qa.py
  2. scripts/preprocess_commonsense_qa.py
EOF
