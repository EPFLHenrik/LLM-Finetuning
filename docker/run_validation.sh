#!/bin/bash
set -euo pipefail
# Submit a Run:AI job to run the local validation inside the course cluster.
# This mirrors the style of docker/download.sh and docker/submit.sh so you can
# run validation on the A100 node with the same environment.
#
# Usage (local quick test):
#   ./scripts/run_validation.sh --local            # run locally (uses --use-tiny by default)
#   ./scripts/run_validation.sh --local --use-tiny
#
# Usage (submit job):
#   ./scripts/run_validation.sh                    # submit job that runs validation using models/qwen3_baseline
#   ./scripts/run_validation.sh my_model_dir       # submit job with custom model dir
#   ./scripts/run_validation.sh --use-tiny         # submit job but use tiny model inside pod (for fast debug)

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_validation.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_validation.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

SUFFIX="validation"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

# Default image with CUDA, PyTorch, and common libraries. Override if needed.
IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Allow quick local run without submitting a job
if [[ "${1:-}" == "--local" ]]; then
  shift || true
  exec python scripts/run_validation.py "$@"
fi

# Parse arguments: optional --use-tiny or model dir
USE_TINY="false"
MODEL_DIR="models/qwen3_thinking"
if [[ "${1:-}" == "--use-tiny" ]]; then
  USE_TINY="true"
elif [[ -n "${1:-}" ]]; then
  MODEL_DIR="$1"
fi

echo ">>> Submitting ${JOB_NAME} (validation job)"

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
    mkdir -p /scratch/hf_cache /scratch/wandb && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd /scratch && \
    git clone https://github.com/$(git config --get remote.origin.url | sed -e 's#.*github.com/##' -e 's#\.git$##') repo || true && \
    cd repo || cd \"${REPO_DIR:-/scratch/workspace/henrik/standard-project-m2-mystery-machine}\" && \
    if [ \"${USE_TINY}\" = \"true\" ]; then \
      python scripts/run_validation.py --use-tiny; \
    else \
      python scripts/run_validation.py --model-dir \"${MODEL_DIR}\"; \
    fi"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF
