#!/bin/bash
# Submit a learnability-filter run as a non-interactive Run:AI job.
#
# Runs `math_task.data.filter_learnability` to score 8 base-model rollouts
# per problem and keep only those whose pass count is in the configurable
# band (default [1, 6] of 8).
#
# The bootstrap regenerates the input JSONL with `prepare_rl.py` on the pod
# if it's not already on /scratch — avoids needing to upload the data file.
#
# Usage (from the repo root):
#   bash math_task/runai/submit_filter.sh smoke    # smoke run with --limit 200
#   bash math_task/runai/submit_filter.sh full     # full 40K run
#   bash math_task/runai/submit_filter.sh smoke2   # custom suffix, smoke defaults
#
# Env overrides:
#   FILTER_INPUT   : JSONL path (default /scratch/user/rl_data/deepscaler_full.jsonl)
#   FILTER_OUTPUT  : filtered JSONL out (default /scratch/user/rl_data/deepscaler_filtered_<suffix>.jsonl)
#   FILTER_LIMIT   : --limit N (default 200 if suffix starts with "smoke", else unset)
#   FILTER_KLO     : --k-lo (default 1)
#   FILTER_KHI     : --k-hi (default 6)
#   FILTER_N       : --n-samples (default 8)
#   FILTER_MAXTOK  : --max-tokens (default 4096)
#   FILTER_ROLLOUTS_OUT : if set, --save-rollouts <path>
#   FILTER_EXTRA_ARGS   : extra args passed to filter_learnability.py

set -euo pipefail

# ============== EDIT IF NEEDED ==============
GASPAR="busato"
GROUP="g02"
REPO_DIR_IN_POD="/scratch/workspace/mattea/standard-project-m2-mystery-machine"
# ============================================

SUFFIX="${1:-smoke}"

# Default --limit based on suffix.
DEFAULT_LIMIT=""
if [[ "${SUFFIX}" == smoke* ]]; then
  DEFAULT_LIMIT="200"
fi

FILTER_INPUT="${FILTER_INPUT:-/scratch/user/rl_data/deepscaler_full.jsonl}"
FILTER_OUTPUT="${FILTER_OUTPUT:-/scratch/user/rl_data/deepscaler_filtered_${SUFFIX}.jsonl}"
FILTER_LIMIT="${FILTER_LIMIT:-$DEFAULT_LIMIT}"
FILTER_KLO="${FILTER_KLO:-1}"
FILTER_KHI="${FILTER_KHI:-6}"
FILTER_N="${FILTER_N:-8}"
FILTER_MAXTOK="${FILTER_MAXTOK:-4096}"
FILTER_ROLLOUTS_OUT="${FILTER_ROLLOUTS_OUT:-}"
FILTER_EXTRA_ARGS="${FILTER_EXTRA_ARGS:-}"

PROJECT="course-cs-552-${GASPAR}"
NODE="a100-40g"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

JOB_NAME="cs552-${GASPAR}-${GROUP}-filter-${SUFFIX}-$(date +%H%M%S)"

# --- Bootstrap wrapper ------------------------------------------------------
# Same git fetch logic as submit_sft.sh. Then:
#   1. install math-verify (used by evaluate.score) if missing
#   2. regenerate the input JSONL if it doesn't exist
#   3. exec the filter script
WRAPPER='set -e
REPO_DIR="'"${REPO_DIR_IN_POD}"'"
git config --global --add safe.directory "$REPO_DIR"
cd "$REPO_DIR"
if [ -n "$GITHUB_TOKEN" ]; then
  echo "[wrapper] pulling latest with token-embedded fetch URL..."
  ORIG_URL=$(git remote get-url origin)
  REPO_PATH=$(echo "$ORIG_URL" | sed -E "s#^https?://([^@]*@)?github\.com/##; s#^git@github\.com:##")
  AUTH_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_PATH}"
  if git fetch "$AUTH_URL" math && git reset --hard FETCH_HEAD; then
    echo "[wrapper] git fetch+reset SUCCESS"
  else
    echo "[wrapper] WARN: git fetch failed; using on-disk code"
  fi
else
  echo "[wrapper] WARN: GITHUB_TOKEN not set; using on-disk code"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__)"

python -c "import math_verify" 2>/dev/null || {
  echo "[wrapper] installing math-verify..."
  pip install --quiet --no-cache-dir "math-verify"
}

if [ ! -f "$FILTER_INPUT" ]; then
  echo "[wrapper] $FILTER_INPUT not found; regenerating with prepare_rl.py"
  python -m math_task.data.prepare_rl --output "$FILTER_INPUT"
fi

exec python -m math_task.data.filter_learnability "$@"'

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Filter args:"
echo "    input        = ${FILTER_INPUT}"
echo "    output       = ${FILTER_OUTPUT}"
echo "    n-samples    = ${FILTER_N}"
echo "    k-lo / k-hi  = ${FILTER_KLO} / ${FILTER_KHI}"
echo "    max-tokens   = ${FILTER_MAXTOK}"
[[ -n "${FILTER_LIMIT}"        ]] && echo "    limit        = ${FILTER_LIMIT}"
[[ -n "${FILTER_ROLLOUTS_OUT}" ]] && echo "    save-rollouts= ${FILTER_ROLLOUTS_OUT}"
[[ -n "${FILTER_EXTRA_ARGS}"   ]] && echo "    extra:       ${FILTER_EXTRA_ARGS}"

FILTER_ARGS=(
  --input  "${FILTER_INPUT}"
  --output "${FILTER_OUTPUT}"
  --n-samples "${FILTER_N}"
  --k-lo "${FILTER_KLO}"
  --k-hi "${FILTER_KHI}"
  --max-tokens "${FILTER_MAXTOK}"
)
[[ -n "${FILTER_LIMIT}"        ]] && FILTER_ARGS+=(--limit "${FILTER_LIMIT}")
[[ -n "${FILTER_ROLLOUTS_OUT}" ]] && FILTER_ARGS+=(--save-rollouts "${FILTER_ROLLOUTS_OUT}")
# shellcheck disable=SC2206
[[ -n "${FILTER_EXTRA_ARGS}"   ]] && FILTER_ARGS+=(${FILTER_EXTRA_ARGS})

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
  --environment GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
  --environment FILTER_INPUT="${FILTER_INPUT}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- bash -c "${WRAPPER}" bash "${FILTER_ARGS[@]}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:   runai logs -f ${JOB_NAME} -p ${PROJECT}
Check status:  runai describe job ${JOB_NAME} -p ${PROJECT}
Cancel:        runai delete job ${JOB_NAME} -p ${PROJECT}

Output JSONL: ${FILTER_OUTPUT}

Smoke job (--limit 200) takes ~10-30 min. Full run (~40K problems) ~10-13h.
The job will run prepare_rl.py if ${FILTER_INPUT} doesn't exist on /scratch.
EOF
