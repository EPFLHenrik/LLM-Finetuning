#!/bin/bash
# Single-evaluation sbatch template.
#
# Reads two environment variables (passed via `sbatch --export=ALL,DATASET=...,LIMIT=...`):
#   DATASET  required, one of {validation, gsm8k, math500}
#   LIMIT    optional, integer cap on number of problems
#
# Override --job-name / --time / partition on the sbatch command line if
# you need different resources per dataset (see submit_all.sh).

#SBATCH --job-name=math_eval
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=math_task/sbatch/logs/%x_%j.out
#SBATCH --error=math_task/sbatch/logs/%x_%j.err

# === EDIT FOR YOUR CLUSTER ===
#SBATCH --partition=gpu
# #SBATCH --account=your-account     # uncomment if required
# #SBATCH --qos=normal                # uncomment if required
# =============================

set -euo pipefail

# Always run from the directory you sbatched from (should be repo root).
cd "${SLURM_SUBMIT_DIR}"

# Activate venv — edit this line if your venv lives elsewhere
# (e.g. /scratch/${USER}/venvs/mnlp)
source .venv/bin/activate

# Keep HF cache off the home quota if your home is small. Uncomment & adjust:
# export HF_HOME="/scratch/${USER}/hf_cache"
# export HF_HUB_CACHE="${HF_HOME}/hub"

# Diagnostics for the log
echo "=================================================================="
echo "[host]    $(hostname)"
echo "[date]    $(date -Iseconds)"
echo "[pwd]     $(pwd)"
echo "[python]  $(which python)  ($(python --version))"
echo "[gpu]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "  (nvidia-smi unavailable)"
echo "[job]     ${SLURM_JOB_NAME} (id=${SLURM_JOB_ID})"
echo "[dataset] ${DATASET:?DATASET env var is required (validation|gsm8k|math500)}"
echo "[limit]   ${LIMIT:-<none>}"
echo "=================================================================="

# Assemble the command — only append --limit if it's set
CMD=(python -m math_task.run_eval --dataset "${DATASET}")
if [[ -n "${LIMIT:-}" ]]; then
    CMD+=(--limit "${LIMIT}")
fi

echo "[exec] ${CMD[*]}"
echo "------------------------------------------------------------------"
"${CMD[@]}"
echo "------------------------------------------------------------------"
echo "[done] exit=$?  $(date -Iseconds)"
