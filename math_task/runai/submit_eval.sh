#!/bin/bash
# Submit ONE math evaluation as a non-interactive Run:AI job.
#
# Mirrors docker/submit.sh exactly (same image, same PVCs, same node pool)
# but runs `python -m math_task.run_eval` instead of Jupyter, and exits
# when done. Use docker/submit.sh for interactive work; this for fire-and-forget.
#
# Usage (from the repo root):
#   bash math_task/runai/submit_eval.sh validation
#   bash math_task/runai/submit_eval.sh math500 50
#   bash math_task/runai/submit_eval.sh math500
#
# NOTE: course quota is 1 A100 per GROUP. If a teammate has a job running,
# yours queues (state: Pending) until they release it.

set -euo pipefail

# ============== EDIT IF NEEDED ==============
GASPAR="busato"
GROUP="g02"
REPO_DIR_IN_POD="/scratch/workspace/${GASPAR}/standard-project-m2-mystery-machine"
# ============================================

DATASET="${1:?usage: submit_eval.sh <dataset> [limit]}"
LIMIT="${2:-}"

PROJECT="course-cs-552-${GASPAR}"
NODE="a100-40g"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

JOB_NAME="cs552-${GASPAR}-${GROUP}-math-${DATASET}-$(date +%H%M%S)"

CMD="python -m math_task.run_eval --dataset ${DATASET}"
[[ -n "${LIMIT}" ]] && CMD="${CMD} --limit ${LIMIT}"

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Pod working dir:  ${REPO_DIR_IN_POD}"
echo ">>> Pod will run:     ${CMD}"

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
  --command -- /bin/bash -lc "\
    set -euo pipefail; \
    mkdir -p /scratch/hf_cache; \
    cd ${REPO_DIR_IN_POD}; \
    git pull --ff-only || echo 'WARN: git pull failed; using on-disk code'; \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
    python -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)'; \
    ${CMD}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:   runai logs -f ${JOB_NAME} -p ${PROJECT}
Check status:  runai describe job ${JOB_NAME} -p ${PROJECT}
Cancel:        runai delete job ${JOB_NAME} -p ${PROJECT}

Results will land in the pod's /scratch (same as your workspace):
   ${REPO_DIR_IN_POD}/math_task/results/

NOTE: course quota is 1 A100 per group. Jobs from teammates queue you.
EOF
