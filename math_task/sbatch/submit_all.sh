#!/bin/bash
# Submit all three math evaluations as a chain:
#   validation (smoke test, 15 min)
#     └── math500 --limit 50 (subset, 45 min, runs ONLY if validation succeeds)
#           └── math500 (full, 6 h, runs ONLY if subset succeeds)
#
# Usage (from the repo root):
#     bash math_task/sbatch/submit_all.sh
#
# To submit them in parallel instead (all three queue independently), pass --parallel:
#     bash math_task/sbatch/submit_all.sh --parallel

set -euo pipefail

# Sanity: must be run from the repo root so SLURM_SUBMIT_DIR is right
if [[ ! -d math_task || ! -d evaluate ]]; then
    echo "ERROR: run this script from the repo root (where math_task/ and evaluate/ live)" >&2
    exit 1
fi

mkdir -p math_task/sbatch/logs

SBATCH_SCRIPT="math_task/sbatch/run_eval.sbatch"
PARALLEL=0
[[ "${1:-}" == "--parallel" ]] && PARALLEL=1

submit() {
    # submit <job_name> <time> <dependency_flag_or_empty> <export_vars>
    local name="$1" time="$2" dep="$3" export_vars="$4"
    local args=(--parsable --job-name="${name}" --time="${time}"
                --export="${export_vars}")
    [[ -n "${dep}" ]] && args+=(--dependency="${dep}")
    sbatch "${args[@]}" "${SBATCH_SCRIPT}"
}

# 1. Smoke test — 10 problems
SMOKE_ID=$(submit "math_validation" "00:15:00" "" \
                  "ALL,DATASET=validation")
echo "[submit] validation   → job ${SMOKE_ID}"

if (( PARALLEL )); then
    DEP_SUBSET=""
    DEP_FULL=""
else
    DEP_SUBSET="afterok:${SMOKE_ID}"
fi

# 2. Quick subset — 50 problems of MATH-500
SUBSET_ID=$(submit "math500_50" "00:45:00" "${DEP_SUBSET}" \
                   "ALL,DATASET=math500,LIMIT=50")
echo "[submit] math500 (50) → job ${SUBSET_ID}  ${DEP_SUBSET:+(after ${SMOKE_ID})}"

if (( ! PARALLEL )); then
    DEP_FULL="afterok:${SUBSET_ID}"
fi

# 3. Full MATH-500 — 500 problems
FULL_ID=$(submit "math500_full" "06:00:00" "${DEP_FULL:-}" \
                 "ALL,DATASET=math500")
echo "[submit] math500 full → job ${FULL_ID}  ${DEP_FULL:+(after ${SUBSET_ID})}"

echo ""
echo "Submitted. Track with:  squeue -u \$USER"
echo "Logs:                   math_task/sbatch/logs/"
echo "Cancel a job:           scancel <id>"
echo "Cancel the whole chain: scancel ${SMOKE_ID} ${SUBSET_ID} ${FULL_ID}"
