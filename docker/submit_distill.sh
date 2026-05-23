#!/bin/bash
# CS-552 — submit a non-interactive distillation job.
#
# This job runs detached from your session.
# The distillation keeps running even after you close your terminal or VPN.
# It is resumable — resubmit with the same DATASETS and OUTPUT_DIR
# and it will skip already-completed examples.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="anastass"
GROUP="g02"
REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"

# Datasets to distill — space-separated, choose from:
#   ceval  milu  mera_mamuramu
# Can also be overridden at call time:
#   DATASETS="milu" ./submit_distill.sh milu
DATASETS="${DATASETS:-ceval milu mera_mamuramu}"

PROCESSED_DIR="${REPO_DIR}/multilingual/data/processed"
OUTPUT_DIR="${REPO_DIR}/multilingual/data/distilled/v7_final"

CONCURRENCY="${CONCURRENCY:-4}"
LOG_EVERY="${LOG_EVERY:-25}"
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit submit_distill.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit submit_distill.sh and set GROUP to your team number, e.g. g02." >&2
    exit 1
fi

if [[ -z "${CSCS_SERVING_API:-}" ]]; then
    echo "ERROR: CSCS_SERVING_API is not set in your local shell." >&2
    echo "Run: export CSCS_SERVING_API='sk-rc-...'" >&2
    exit 1
fi

GPUS="${GPUS:-1}"
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-distill}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

LOG_FILE="${OUTPUT_DIR}/distill_${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Datasets:      ${DATASETS}"
echo ">>> Processed dir: ${PROCESSED_DIR}"
echo ">>> Output dir:    ${OUTPUT_DIR}"
echo ">>> Log file:      ${LOG_FILE}"
echo ">>> Concurrency:   ${CONCURRENCY}"
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
  --environment CSCS_SERVING_API="${CSCS_SERVING_API}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache ${OUTPUT_DIR} && \
    cd ${REPO_DIR} && \
    pip install httpx --break-system-packages -q && \
    echo '[distill-pod] Started at '\$(date) | tee -a ${LOG_FILE} && \
    echo '[distill-pod] Datasets: ${DATASETS}' | tee -a ${LOG_FILE} && \
    echo '[distill-pod] Processed dir: ${PROCESSED_DIR}' | tee -a ${LOG_FILE} && \
    echo '[distill-pod] Output dir: ${OUTPUT_DIR}' | tee -a ${LOG_FILE} && \
    python -m multilingual.distillation.distill \
      --datasets ${DATASETS} \
      --processed_dir ${PROCESSED_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --concurrency ${CONCURRENCY} \
      --log_every ${LOG_EVERY} \
      2>&1 | tee -a ${LOG_FILE} && \
    echo '[distill-pod] Finished at '\$(date) | tee -a ${LOG_FILE}"

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

Tail the log file directly once the pod is running:
  runai bash ${JOB_NAME} -p ${PROJECT}
  tail -f ${LOG_FILE}

Stop the job:
  runai workload delete ${JOB_NAME} -p ${PROJECT}

NOTE:
  This is a non-interactive job. It keeps running after you disconnect.
  The distillation is resumable. If the job is killed, resubmit with the
  same DATASETS and OUTPUT_DIR and it will skip already-done examples.

EOF