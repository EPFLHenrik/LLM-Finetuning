#!/bin/bash
# CS-552 — submit a non-interactive distillation job.
#
# Unlike submit.sh (Jupyter Lab), this job runs detached from your session.
# The distillation keeps running even after you close your terminal or VPN.
# It is also resumable — resubmit with the same DATASETS and OUTPUT_DIR
# and it will skip already-completed examples.
#
# ============================================================
#  STEP 1  — set your Gaspar username, team group ID, and
#            distillation settings below (all REQUIRED)
# ============================================================
#
# Usage:
#   ./submit_distill.sh               # default suffix "distill"
#   ./submit_distill.sh mera          # custom suffix → job name includes "mera"
#
# To run 3 parallel jobs (one per dataset group):
#   DATASETS="mera_mamuramu mera_mathlogicqa" ./submit_distill.sh mera
#   DATASETS="milu"                           ./submit_distill.sh milu
#   DATASETS="ceval"                          ./submit_distill.sh ceval
#
# Monitor after submission:
#   runai workload describe <job-name> -p <project>
#   runai workload logs <job-name> -p <project>     (stream logs)
#   runai workload list -p <project>
#
# Shell into the pod:
#   runai bash <job-name> -p <project>
#   tail -f <LOG_FILE>                               (defined below)
#
# Attach with VS Code:
#   refresh the Kubernetes sidebar and attach to pod <job-name>-0-0
#
# Stop the job:
#   runai workload delete <job-name> -p <project>

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="anastass"
GROUP="g02"
REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"

# Datasets to distill — space-separated, choose from:
#   mera_mamuramu  mera_mathlogicqa  milu  ceval  global_mmlu
# Can also be overridden at call time: DATASETS="milu" ./submit_distill.sh milu
DATASETS="${DATASETS:-mera_mamuramu mera_mathlogicqa}"

SFT_DIR="${REPO_DIR}/multilingual/data/sft/v7"
OUTPUT_DIR="${REPO_DIR}/multilingual/data/distilled/v7"

CONCURRENCY=4     # parallel API calls
LOG_EVERY=25      # print progress every N examples
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit submit_distill.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit submit_distill.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set in your shell. Run: export HF_TOKEN=..." >&2
    exit 1
fi

if [[ -z "${CSCS_SERVING_API:-}" ]]; then
    echo "ERROR: CSCS_SERVING_API is not set. Run: export CSCS_SERVING_API=sk-rc-..." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-distill}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Log file written inside the cluster (readable with runai bash + tail -f)
LOG_FILE="${OUTPUT_DIR}/distill_${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Datasets:   ${DATASETS}"
echo ">>> SFT dir:    ${SFT_DIR}"
echo ">>> Output dir: ${OUTPUT_DIR}"
echo ">>> Log file:   ${LOG_FILE}"
echo ">>> (non-interactive — keeps running after disconnect)"

MSYS_NO_PATHCONV=1 runai training submit "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu-devices-request "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN}" \
  --environment CSCS_SERVING_API="${CSCS_SERVING_API}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache ${OUTPUT_DIR} && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd ${REPO_DIR} && \
    pip install httpx --break-system-packages -q && \
    echo '[distill-pod] Ready at \$(date)' && \
    echo '[distill-pod] Attach with VS Code or runai bash, then start tmux sessions.' && \
    sleep infinity"
cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:
  runai workload describe ${JOB_NAME} -p ${PROJECT}

List workloads:
  runai workload list -p ${PROJECT}

Stream logs:
  runai workload logs ${JOB_NAME} -p ${PROJECT}

If streaming logs with the new command is not supported, use:
  runai logs -f ${JOB_NAME} -p ${PROJECT}

Tail the log file directly (once pod is running):
  runai bash ${JOB_NAME} -p ${PROJECT}
  tail -f ${LOG_FILE}

Attach with VS Code:
  refresh the Kubernetes sidebar and attach to pod ${JOB_NAME}-0-0

Stop the job:
  runai workload delete ${JOB_NAME} -p ${PROJECT}

NOTE: This is a non-interactive job. It keeps running after you disconnect.
      The distillation is resumable — if the job is killed, resubmit with
      the same DATASETS and OUTPUT_DIR and it will skip already-done examples.

Run:AI note:
  If you see a transient [PodGrouperWarning] that says the PodGroup
  "object has been modified", wait and run the describe command again.
  This is a scheduler reconciliation warning, not an error. The job is
  only stuck if it remains Pending after a few minutes.

Storage inside the pod:
  /scratch             team scratch — your group's primary workspace, RW
  /shared-ro/datasets  course datasets (read-only)
  /shared-ro/models    course models   (read-only)
  /shared-rw           shared with ALL students — be careful what you write

EOF