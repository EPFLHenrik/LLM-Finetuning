#!/bin/bash
# Submit a CISPO LoRA RL training run as a non-interactive Run:AI job.
#
# Mirrors submit_sft.sh. The two differences:
#  1. Bootstrap pip-installs ms-swift (not in the course Docker image).
#  2. Entry is math_task/training/train_grpo.sh (the CISPO wrapper),
#     not train_sft.py.
#
# Usage (from the repo root):
#   bash math_task/runai/submit_rl.sh             # phase 4k pilot, v1
#   bash math_task/runai/submit_rl.sh v2          # phase 4k pilot, v2
#   PHASE=8k bash math_task/runai/submit_rl.sh v1
#
# Env overrides (passed through to the pod):
#   PHASE       : 4k | 8k | 16k                (default 4k)
#   PILOT       : 1 for the 100-step smoke test, 0 for full run (default 1)
#   RL_DATA     : input JSONL (default /scratch/user/rl_data/deepscaler_filtered.jsonl)
#   RL_OUTPUT   : ckpt dir   (default /scratch/user/ckpt/math_rl_<phase>_v1)
#   RL_MODEL    : --model path. Default Qwen/Qwen3-1.7B (Track B from-base).
#                 Set to cs-552-2026-mystery-machine/math_model for Track A
#                 (already-merged SFT-200 + new system-prompt chat template).
#   INIT_FROM   : path to previous-phase LoRA adapter (default: none = from base)
#   MAX_STEPS   : override step count
#   LR          : learning rate            (default 1e-6)
#   KL_BETA     : KL coefficient            (default 0.001)
#   EPS_HIGH    : CISPO ε_high              (default 5.0)
#   GRAD_ACCUM  : gradient accumulation     (default 8)
#   RL_EXTRA_ARGS : extra args appended to swift rlhf

set -euo pipefail

# ============== EDIT IF NEEDED ==============
GASPAR="busato"
GROUP="g02"
REPO_DIR_IN_POD="/scratch/workspace/mattea/standard-project-m2-mystery-machine"
# ============================================

SUFFIX="${1:-v1}"

PHASE="${PHASE:-4k}"
PILOT="${PILOT:-1}"
RL_DATA="${RL_DATA:-/scratch/user/rl_data/deepscaler_filtered.jsonl}"
RL_OUTPUT="${RL_OUTPUT:-/scratch/user/ckpt/math_rl_${PHASE}_${SUFFIX}}"
RL_MODEL="${RL_MODEL:-}"
INIT_FROM="${INIT_FROM:-}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-}"
KL_BETA="${KL_BETA:-}"
EPS_HIGH="${EPS_HIGH:-}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
RL_EXTRA_ARGS="${RL_EXTRA_ARGS:-}"

PROJECT="course-cs-552-${GASPAR}"
NODE="a100-40g"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

JOB_NAME="cs552-${GASPAR}-${GROUP}-rl-${PHASE}-${SUFFIX}-$(date +%H%M%S)"
# k8s job names are RFC 1123: lowercase alphanumeric + '-' only. Sanitize any
# underscores/uppercase from SUFFIX so e.g. "8k_v1" doesn't fail the submit.
JOB_NAME=$(echo "$JOB_NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')

# --- Bootstrap wrapper ------------------------------------------------------
# Mirrors submit_sft.sh's token-embedded fetch, then ALSO pip-installs ms-swift
# (not in the course image) before exec'ing the trainer wrapper.
#
# We pin ms-swift to >=3.0 since CISPO support landed in 3.0+.
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

echo "[wrapper] installing ms-swift (>=3.0)..."
pip install --quiet --no-cache-dir "ms-swift>=3.0" math-verify "bitsandbytes>=0.43" || {
  echo "[wrapper] ERROR: pip install failed" >&2
  exit 1
}
# Verify swift is importable (no print with brackets/quotes — those get
# mangled through the laptop->runai->bash-c->python-c quoting layers and
# crash the wrapper under set -e).
python -c "import swift"

exec bash math_task/training/train_grpo.sh "$@"'

echo ">>> Submitting ${JOB_NAME}"
echo ">>> RL args:"
echo "    phase     = ${PHASE}"
echo "    pilot     = ${PILOT}"
echo "    model     = ${RL_MODEL:-(default: Qwen/Qwen3-1.7B)}"
echo "    dataset   = ${RL_DATA}"
echo "    output    = ${RL_OUTPUT}"
echo "    init_from = ${INIT_FROM:-(none)}"
[[ -n "${MAX_STEPS}"  ]] && echo "    max_steps  = ${MAX_STEPS}"
[[ -n "${LR}"         ]] && echo "    lr         = ${LR}"
[[ -n "${KL_BETA}"    ]] && echo "    kl_beta    = ${KL_BETA}"
[[ -n "${EPS_HIGH}"   ]] && echo "    eps_high   = ${EPS_HIGH}"
[[ -n "${GRAD_ACCUM}" ]] && echo "    grad_accum = ${GRAD_ACCUM}"
[[ -n "${RL_EXTRA_ARGS}" ]] && echo "    extra:    ${RL_EXTRA_ARGS}"

# train_grpo.sh only parses --dataset and --output as CLI args; everything
# else for `swift rlhf` comes from the EXTRA_ARGS *env var* (it does
# `swift rlhf ... $EXTRA_ARGS`). So RL_EXTRA_ARGS must be forwarded as the
# EXTRA_ARGS env var, NOT appended to TRAIN_ARGS (which would make
# train_grpo.sh's parser choke on e.g. --model_type).
TRAIN_ARGS=(
  --dataset "${RL_DATA}"
  --output  "${RL_OUTPUT}"
)

# Env vars we explicitly forward to the pod (in addition to the standard ones).
ENV_FORWARDS=(
  --environment "PHASE=${PHASE}"
  --environment "PILOT=${PILOT}"
)
[[ -n "${RL_EXTRA_ARGS}" ]] && ENV_FORWARDS+=(--environment "EXTRA_ARGS=${RL_EXTRA_ARGS}")
[[ -n "${RL_MODEL}"   ]] && ENV_FORWARDS+=(--environment "RL_MODEL=${RL_MODEL}")
[[ -n "${INIT_FROM}"  ]] && ENV_FORWARDS+=(--environment "INIT_FROM=${INIT_FROM}")
[[ -n "${MAX_STEPS}"  ]] && ENV_FORWARDS+=(--environment "MAX_STEPS=${MAX_STEPS}")
[[ -n "${LR}"         ]] && ENV_FORWARDS+=(--environment "LR=${LR}")
[[ -n "${KL_BETA}"    ]] && ENV_FORWARDS+=(--environment "KL_BETA=${KL_BETA}")
[[ -n "${EPS_HIGH}"   ]] && ENV_FORWARDS+=(--environment "EPS_HIGH=${EPS_HIGH}")
[[ -n "${GRAD_ACCUM}" ]] && ENV_FORWARDS+=(--environment "GRAD_ACCUM=${GRAD_ACCUM}")
[[ -n "${NUM_GEN}"    ]] && ENV_FORWARDS+=(--environment "NUM_GEN=${NUM_GEN}")

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
  --environment WANDB_PROJECT=cs552-math-rl \
  "${ENV_FORWARDS[@]}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- bash -c "${WRAPPER}" bash "${TRAIN_ARGS[@]}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Stream logs:   runai logs -f ${JOB_NAME} -p ${PROJECT}
Check status:  runai describe job ${JOB_NAME} -p ${PROJECT}
Cancel:        runai delete job ${JOB_NAME} -p ${PROJECT}

Checkpoints will land in: ${RL_OUTPUT}/checkpoint-*

CISPO phase = ${PHASE}, pilot = ${PILOT}.
Note: pod will pip-install ms-swift on startup (~30s overhead).
EOF
