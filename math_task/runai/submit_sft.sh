#!/bin/bash
# Submit an SFT training run as a non-interactive Run:AI job.
#
# Interactive pods get suspended after idle timeout (~hours), killing any
# running training. Non-interactive `Train`-type jobs run unattended until
# the training script exits — perfect for the ~6-hour SFT run.
#
# Usage (from the repo root):
#   bash math_task/runai/submit_sft.sh
#   bash math_task/runai/submit_sft.sh v2     # custom suffix
#
# Optional env vars (passed through to the pod):
#   SFT_DATA       : JSONL path (default /scratch/user/sft_data/openr1_math_5k.jsonl)
#   SFT_OUTPUT     : checkpoint dir (default /scratch/user/ckpt/math_sft_v1)
#   SFT_EPOCHS     : default 2
#   SFT_SAVE_STEPS : default 50
#   SFT_EVAL_STEPS : default 50
#   SFT_EVAL_FRAC  : default 0.05
#   SFT_EXTRA_ARGS : extra args passed verbatim to train_sft.py
#
# Example with overrides:
#   SFT_EPOCHS=3 SFT_SAVE_STEPS=100 bash math_task/runai/submit_sft.sh v2

set -euo pipefail

# ============== EDIT IF NEEDED ==============
GASPAR="busato"
GROUP="g02"
REPO_DIR_IN_POD="/scratch/workspace/mattea/standard-project-m2-mystery-machine"
# ============================================

SUFFIX="${1:-v1}"

SFT_DATA="${SFT_DATA:-/scratch/user/sft_data/openr1_math_5k.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-/scratch/user/ckpt/math_sft_${SUFFIX}}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-50}"
SFT_EVAL_FRAC="${SFT_EVAL_FRAC:-0.05}"
SFT_EXTRA_ARGS="${SFT_EXTRA_ARGS:-}"

PROJECT="course-cs-552-${GASPAR}"
NODE="a100-40g"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

JOB_NAME="cs552-${GASPAR}-${GROUP}-sft-${SUFFIX}-$(date +%H%M%S)"

# --- Bootstrap wrapper ------------------------------------------------------
# Same pattern as submit_eval.sh: install git credentials from GITHUB_TOKEN,
# safe-directory the repo, pull latest, then exec the training command.
WRAPPER='set -e
REPO_DIR="'"${REPO_DIR_IN_POD}"'"
git config --global --add safe.directory "$REPO_DIR"
cd "$REPO_DIR"
if [ -n "$GITHUB_TOKEN" ]; then
  echo "[wrapper] pulling latest via http.extraheader..."
  if git -c "http.https://github.com/.extraheader=Authorization: Bearer $GITHUB_TOKEN" pull --ff-only; then
    echo "[wrapper] git pull SUCCESS"
  else
    echo "[wrapper] WARN: git pull failed; using on-disk code"
  fi
else
  echo "[wrapper] WARN: GITHUB_TOKEN not set; using on-disk code"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, vllm, transformers, trl, peft; print(f\"torch={torch.__version__}  vllm={vllm.__version__}  transformers={transformers.__version__}  trl={trl.__version__}  peft={peft.__version__}\")"

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

# Build the python args. Use an array so quoting is unambiguous.
TRAIN_ARGS=(
    --data "${SFT_DATA}"
    --output "${SFT_OUTPUT}"
    --epochs "${SFT_EPOCHS}"
    --save-steps "${SFT_SAVE_STEPS}"
    --eval-steps "${SFT_EVAL_STEPS}"
    --eval-frac "${SFT_EVAL_FRAC}"
)
# Append extra args (word-split intentional here).
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
Job will exit on its own when training finishes (no idle timeout).

NOTE: course quota is 1 A100 per group. If you still have an interactive
pod holding the GPU, delete it so this training job can run.
EOF
