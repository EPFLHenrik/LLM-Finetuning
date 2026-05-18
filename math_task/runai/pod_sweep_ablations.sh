#!/bin/bash
# Pod-side: loop the 6 ablation cells on a single GPU reservation.
#
#   prompt_variant  ∈ {none, boxed, classify}
#   thinking        ∈ {on, off}     (Qwen3 enable_thinking)
#
# Each cell runs run_eval.py and writes its own stem-prefixed files into
# math_task/results/, so nothing clobbers across cells. Model loads once per
# cell (vLLM doesn't make sharing trivial across separate python procs);
# ~30-60s overhead × 6 ≈ 3-6 min total, negligible vs generation time.
#
# Usage (on the pod, invoked by sweep_ablations.sh):
#   bash math_task/runai/pod_sweep_ablations.sh <dataset> [limit]
# Example:
#   bash math_task/runai/pod_sweep_ablations.sh math500 100

set -euo pipefail

DATASET="${1:?usage: pod_sweep_ablations.sh <dataset> [limit]}"
LIMIT="${2:-}"

REPO_DIR="/scratch/workspace/mattea/standard-project-m2-mystery-machine"

git config --global --add safe.directory "${REPO_DIR}"
mkdir -p /scratch/hf_cache

cd "${REPO_DIR}"
git pull --ff-only || echo "WARN: git pull failed; using on-disk code"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)'

VARIANTS=(none boxed classify)
THINKINGS=(on off)

START_TS=$(date +%s)
for VARIANT in "${VARIANTS[@]}"; do
  for THINKING in "${THINKINGS[@]}"; do
    echo
    echo "================================================================"
    echo "[sweep] cell: dataset=${DATASET} variant=${VARIANT} thinking=${THINKING} limit=${LIMIT:-full}"
    echo "================================================================"
    CELL_START=$(date +%s)

    CMD=(python -m math_task.run_eval
         --dataset "${DATASET}"
         --prompt-variant "${VARIANT}"
         --thinking "${THINKING}")
    [[ -n "${LIMIT}" ]] && CMD+=(--limit "${LIMIT}")
    echo "[exec] ${CMD[*]}"
    "${CMD[@]}"

    echo "[sweep] cell finished in $(( $(date +%s) - CELL_START ))s"
  done
done

echo
echo "[sweep] all 6 cells done in $(( $(date +%s) - START_TS ))s"
echo "[sweep] results in: ${REPO_DIR}/math_task/results/"
ls -1 "${REPO_DIR}/math_task/results/" | grep -E "^${DATASET}_.*_scored\.json$" || true
