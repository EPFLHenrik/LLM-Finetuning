#!/bin/bash
# Run v19_15k, then v19_20k sequentially.
# Run with:
#   nohup bash nohup/train_v19_both.sh > logs/train_v19_both.log 2>&1 &

set -euo pipefail

REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"
cd "${REPO_DIR}"

mkdir -p logs

echo "========================================"
echo "Starting chained v19 trainings"
echo "Time: $(date)"
echo "========================================"

echo ""
echo "========================================"
echo "RUN 1/2: v19_15k"
echo "========================================"
bash nohup/train_v19_15k.sh

echo ""
echo "========================================"
echo "Finished v19_15k"
echo "Time: $(date)"
echo "========================================"

echo ""
echo "========================================"
echo "RUN 2/2: v19_20k"
echo "========================================"
bash nohup/train_v19_20k.sh

echo ""
echo "========================================"
echo "Finished all chained v19 trainings"
echo "Time: $(date)"
echo "========================================"