#!/bin/bash
# Run v19 augmented trainings sequentially.
# Run with:
#   nohup bash nohup/train_v19_augmented_remaining.sh > logs/train_v19_augmented_all.log 2>&1 &

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
echo "RUN 1/3: v19_15k augmented"
echo "========================================"
bash nohup/train_v19_15k_augmented.sh

echo ""
echo "========================================"
echo "RUN 2/3: v19_20k augmented"
echo "========================================"
bash nohup/train_v19_20k_augmented.sh

echo ""
echo "========================================"
echo "Finished v19 augmented trainings"
echo "Time: $(date)"
echo "========================================"

echo ""
echo "========================================"
echo "RUN 3/3: v19_20k (additional, non-augmented)"
echo "========================================"
bash nohup/train_v19_20k.sh


echo ""
echo "========================================"
echo "Finished all chained v19 trainings"
echo "Time: $(date)"
echo "========================================"