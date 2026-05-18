#!/bin/bash
# Pod entrypoint for batch math evaluations.
#
# Run via runai by passing this script as the --command, e.g.
#   bash /scratch/.../math_task/runai/pod_entrypoint.sh math500 50
#
# Optional ablation knobs are read from env vars (set via runai --environment
# in submit_eval.sh) so the positional signature stays stable:
#   VARIANT   : none|boxed|classify   (default: none)
#   THINKING  : on|off                (default: on)
#
# Designed so the runai submit wrapper just shells out to this file —
# avoids fragile multi-line --command strings.

set -euo pipefail

DATASET="${1:?usage: pod_entrypoint.sh <dataset> [limit]}"
LIMIT="${2:-}"
VARIANT="${VARIANT:-none}"
THINKING="${THINKING:-on}"

REPO_DIR="/scratch/workspace/mattea/standard-project-m2-mystery-machine"

# Trust the repo so git stops complaining about /scratch ownership.
git config --global --add safe.directory "${REPO_DIR}"

# HF cache lives on /scratch so successive pods reuse downloads.
mkdir -p /scratch/hf_cache

cd "${REPO_DIR}"

# Try to pull, but fall back to on-disk code if there's no usable token.
git pull --ff-only || echo "WARN: git pull failed; using on-disk code"

# Sanity prints — first thing visible in the runai log.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)'

# Build and run the actual eval.
CMD=(python -m math_task.run_eval
     --dataset "${DATASET}"
     --prompt-variant "${VARIANT}"
     --thinking "${THINKING}")
[[ -n "${LIMIT}" ]] && CMD+=(--limit "${LIMIT}")
echo "[exec] ${CMD[*]}"
"${CMD[@]}"