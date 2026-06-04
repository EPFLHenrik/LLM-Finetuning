#!/bin/bash
# Run v19 augmented trainings sequentially.
# Run with:
#   nohup bash nohup/train_v19_augmented_all.sh > logs/train_v19_augmented_all.log 2>&1 &

set -euo pipefail

REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"
cd "${REPO_DIR}"

mkdir -p logs

echo "========================================"
echo "Starting v19 augmented trainings"
echo "Time: $(date)"
echo "========================================"

echo ""
echo "========================================"
echo "RUN 1/2: v19_15k augmented"
echo "========================================"
bash nohup/train_v19_15k_augmented.sh

echo ""
echo "========================================"
echo "RUN 2/2: v19_20k augmented"
echo "========================================"
bash nohup/train_v19_20k_augmented.sh

echo ""
echo "========================================"
echo "Finished v19 augmented trainings"
echo "Time: $(date)"
echo "========================================"