#!/bin/bash
# Submit ONE math evaluation as a non-interactive Run:AI job.
#
# The pod runs `pod_entrypoint.sh <dataset> [limit]` which lives in the repo.
# Keeping the bash logic in a real file (not embedded in this string) avoids
# zsh/runai quote-handling bugs.
#
# Usage (from the repo root):
#   bash math_task/runai/submit_eval.sh validation
#   bash math_task/runai/submit_eval.sh math500 50
#   bash math_task/runai/submit_eval.sh math500
#
# Optional ablation knobs (env vars; passed through to pod_entrypoint.sh):
#   VARIANT=boxed   : user-prompt prefix (none|boxed|classify; default none)
#   THINKING=off    : Qwen3 enable_thinking (on|off; default on)
#   BUDGET=ci       : token budget (ci=4096 | final=16384; default final)
# Example:
#   BUDGET=final bash math_task/runai/submit_eval.sh math500_stratified
#   VARIANT=boxed THINKING=off BUDGET=ci bash math_task/runai/submit_eval.sh math500 100
set -euo pipefail

# ============== EDIT IF NEEDED ==============
GASPAR="busato"
GROUP="g02"
REPO_DIR_IN_POD="/scratch/workspace/mattea/standard-project-m2-mystery-machine"
# ============================================

DATASET="${1:?usage: submit_eval.sh <dataset> [limit]}"
LIMIT="${2:-}"
VARIANT="${VARIANT:-none}"
THINKING="${THINKING:-on}"
BUDGET="${BUDGET:-final}"

PROJECT="course-cs-552-${GASPAR}"
NODE="a100-40g"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Run:AI/k8s job names: lowercase alphanumeric + '-' only, AND ≤63 chars total
# (runai appends a `-0-0` pod suffix on top). Sanitize underscores and
# abbreviate long dataset slugs to stay under the limit. The original DATASET
# string is still passed to pod_entrypoint.sh unchanged.
DATASET_SLUG="${DATASET//_/-}"
DATASET_SLUG="${DATASET_SLUG//stratified/strat}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${DATASET_SLUG}-${VARIANT}-th${THINKING}-b${BUDGET}-$(date +%H%M%S)"
ENTRYPOINT="${REPO_DIR_IN_POD}/math_task/runai/pod_entrypoint.sh"

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Pod will run: VARIANT=${VARIANT} THINKING=${THINKING} BUDGET=${BUDGET} bash ${ENTRYPOINT} ${DATASET} ${LIMIT}"

# Build the pod command as a bash array — safer than interpolating into a string.
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
  --environment VARIANT="${VARIANT}" \
  --environment THINKING="${THINKING}" \
  --environment BUDGET="${BUDGET}" \
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

NOTE: course quota is 1 A100 per group.
EOF