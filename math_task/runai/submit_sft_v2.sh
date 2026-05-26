#!/bin/bash
# Submit the SFT v2 training run as a non-interactive Run:AI job.
#
# Differences from submit_sft.sh:
#  - Default --data points at the SFT v2 dataset (system-prompted, ~30K rows).
#  - Default --output is /scratch/user/ckpt/math_sft_v2.
#  - Slightly larger save cadence (200 steps) since the run is longer.
#
# Usage (from the repo root):
#   bash math_task/runai/submit_sft_v2.sh           # default suffix "v2"
#   bash math_task/runai/submit_sft_v2.sh v2_run2   # custom suffix
#
# Optional env vars (passed through to the pod):
#   SFT_DATA       : JSONL path (default /scratch/user/sft_data/sft_v2_30k.jsonl)
#   SFT_OUTPUT     : ckpt dir   (default /scratch/user/ckpt/math_sft_<suffix>)
#   SFT_EPOCHS     : default 2
#   SFT_SAVE_STEPS : default 200
#   SFT_EVAL_STEPS : default 200
#   SFT_EVAL_FRAC  : default 0.02
#   SFT_EXTRA_ARGS : extra args appended verbatim to train_sft.py
#
# Example:
#   SFT_EPOCHS=3 bash math_task/runai/submit_sft_v2.sh v2_3ep
set -euo pipefail

# ============== EDIT IF NEEDED ==============
GASPAR="busato"
GROUP="g02"
REPO_DIR_IN_POD="/scratch/workspace/mattea/standard-project-m2-mystery-machine"
# ============================================

SUFFIX="${1:-v2}"

SFT_DATA="${SFT_DATA:-/scratch/user/sft_data/sft_v2_30k.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-/scratch/user/ckpt/math_sft_${SUFFIX}}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-200}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-200}"
SFT_EVAL_FRAC="${SFT_EVAL_FRAC:-0.02}"
SFT_EXTRA_ARGS="${SFT_EXTRA_ARGS:-}"

PROJECT="course-cs-552-${GASPAR}"
NODE="a100-40g"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

JOB_NAME="cs552-${GASPAR}-${GROUP}-sft-${SUFFIX}-$(date +%H%M%S)"

# --- Bootstrap wrapper ------------------------------------------------------
# Same token-embedded fetch + safe-directory dance as submit_sft.sh.
# See submit_sft.sh for the rationale (history of failed credential helpers).
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
python -c "import torch, vllm, transformers, trl, peft; print(torch.__version__, vllm.__version__, transformers.__version__, trl.__version__, peft.__version__)"

# Upgrade bitsandbytes to a CUDA-12.8-compatible build. peft loads LoRA via
# `from .bnb import dispatch_bnb_8bit`; the stock course-image bnb is pre-12.8
# and crashes during the unconditional import. Fixed by 0.45+. We skip the
# version print here because nested shell quoting (laptop -> runai -> pod
# -> bash -c -> python -c) eats escaped quotes and breaks the line; if the
# install fails, pip itself exits non-zero and set -e aborts.
echo "[wrapper] upgrading bitsandbytes for cu128..."
pip install -U --quiet bitsandbytes

exec python -m math_task.training.train_sft "$@"'

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Training args:"
echo "    --data        ${SFT_DATA}"
echo "    --output      ${SFT_OUTPUT}"
echo "    --epochs      ${SFT_EPOCHS}"
echo "    --save-steps  ${SFT_SAVE_STEPS}"
echo "    --eval-steps  ${SFT_EVAL_STEPS}"
echo "    --eval-frac   ${SFT_EVAL_FRAC}"
[[ -n "${SFT_EXTRA_ARGS}" ]] && echo "    extra:        ${SFT_EXTRA_ARGS}"

TRAIN_ARGS=(
    --data "${SFT_DATA}"
    --output "${SFT_OUTPUT}"
    --epochs "${SFT_EPOCHS}"
    --save-steps "${SFT_SAVE_STEPS}"
    --eval-steps "${SFT_EVAL_STEPS}"
    --eval-frac "${SFT_EVAL_FRAC}"
)
# shellcheck disable=SC2206
[[ -n "${SFT_EXTRA_ARGS}" ]] && TRAIN_ARGS+=(${SFT_EXTRA_ARGS})

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
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment WANDB_DIR=/scratch/wandb \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- bash -c "${WRAPPER}" bash "${TRAIN_ARGS[@]}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:   runai logs -f ${JOB_NAME} -p ${PROJECT}
Check status:  runai describe job ${JOB_NAME} -p ${PROJECT}
Cancel:        runai delete job ${JOB_NAME} -p ${PROJECT}

Checkpoints will land in: ${SFT_OUTPUT}/checkpoint-*
Wallclock estimate: ~6-8h on ~30K rows × 2 epochs at max_length=8192.

NOTE: course quota is 1 A100 per group. Delete any interactive pod first.
EOF
