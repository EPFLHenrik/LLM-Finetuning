#!/bin/bash
# Submit ONE runai job that runs the full 6-cell ablation sweep on the pod:
#   {none, boxed, classify}  ×  thinking {on, off}
#
# Single GPU reservation, sequential cells. Cleaner than 6 queued jobs because
# the team only has 1 A100 quota and we get one log to stream.
#
# Usage (from repo root):
#   bash math_task/runai/sweep_ablations.sh math500 100
#   bash math_task/runai/sweep_ablations.sh math500          # full 500
#   bash math_task/runai/sweep_ablations.sh validation       # 10-sample smoke
set -euo pipefail

# ============== EDIT IF NEEDED ==============
GASPAR="busato"
GROUP="g02"
REPO_DIR_IN_POD="/scratch/workspace/mattea/standard-project-m2-mystery-machine"
# ============================================

DATASET="${1:?usage: sweep_ablations.sh <dataset> [limit]}"
LIMIT="${2:-}"

PROJECT="course-cs-552-${GASPAR}"
NODE="a100-40g"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

JOB_NAME="cs552-${GASPAR}-${GROUP}-math-sweep-${DATASET}-$(date +%H%M%S)"
ENTRYPOINT="${REPO_DIR_IN_POD}/math_task/runai/pod_sweep_ablations.sh"

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Pod will run: bash ${ENTRYPOINT} ${DATASET} ${LIMIT}"

POD_CMD=(bash "${ENTRYPOINT}" "${DATASET}")
[[ -n "${LIMIT}" ]] && POD_CMD+=("${LIMIT}")

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu 1 \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir "${REPO_DIR_IN_POD}" \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- "${POD_CMD[@]}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:   runai logs -f ${JOB_NAME} -p ${PROJECT}
Check status:  runai describe job ${JOB_NAME} -p ${PROJECT}
Cancel:        runai delete job ${JOB_NAME} -p ${PROJECT}

Results land in: ${REPO_DIR_IN_POD}/math_task/results/
File-stem pattern: ${DATASET}_<safe_model>_<variant>_think-<on|off>_{raw,gens,scored}.{json,jsonl}

NOTE: course quota is 1 A100 per group. This single job runs all 6 cells
sequentially (~3-6 min model-load overhead total, dominated by generation).
EOF
