#!/bin/bash
set -euo pipefail
# Submit a Run:AI job to prepare and optionally push the baseline checkpoint.
# Keeps all work under /scratch/workspace/henrik/standard-project-m2-mystery-machine.
#
# Usage (local):
#   ./docker/run_setup_baseline.sh --local [--repo <REPO>] [--dest <dest_rel>] [--push-repo <hf_repo>] [--commit-message "..."] --yes
#
# Usage (submit job):
#   ./docker/run_setup_baseline.sh                       # submit job that runs setup_baseline.py
#   ./docker/run_setup_baseline.sh --push-repo user/model # also push to HF model repo

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_setup_baseline.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_setup_baseline.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

WORKSPACE_BASE="/scratch/workspace/henrik/standard-project-m2-mystery-machine"
REPO_DIR="${WORKSPACE_BASE}"
GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="setup-baseline"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Allow quick local run without submitting a job
if [[ "${1:-}" == "--local" ]]; then
  shift || true
  exec python scripts/setup_baseline.py "$@"
fi

# Configure args from environment variables (mirrors download.sh style)
# Environment variables you can set before running this script:
#  DEST_REL (relative path under WORKSPACE_BASE, default: models/qwen3_baseline)
#  SOURCE_REPO (HF source repo, default: Qwen/Qwen3-1.7B)
#  PUSH_REPO (HF repo id to push to; optional)
#  COMMIT_MESSAGE (commit message when pushing)
#  YES (set to "true" to auto-confirm overwrites)

DEST_REL="${DEST_REL:-models/qwen3_baseline}"
SOURCE_REPO="${SOURCE_REPO:-Qwen/Qwen3-1.7B}"
# Default HF repo to push prepared artifacts to (can be overridden by env)
PUSH_REPO="${PUSH_REPO:-cs-552-2026-mystery-machine/general_knowledge_model}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Add model artifacts from setup_baseline}"
YES_FLAG="${YES:-false}"

# Normalize dest to be relative and safe
DEST_REL="${DEST_REL#/}"
if [[ "$DEST_REL" == *".."* ]]; then
  echo "Refusing dest containing '..'" >&2
  exit 1
fi
DEST_ABS="${WORKSPACE_BASE}/${DEST_REL}"

echo ">>> Submitting ${JOB_NAME} to prepare baseline (dest: ${DEST_ABS})"

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
  python3 scripts/setup_baseline.py \
  --repo "${SOURCE_REPO}" \
  --push-repo "${PUSH_REPO}" \
  --commit-message "${COMMIT_MESSAGE}" \
  --yes
cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF
