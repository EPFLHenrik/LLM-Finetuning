#!/bin/bash
# eval_loop.sh - runs eval_merged.sh repeatedly until stopped
set -euo pipefail

LOGS="$REPO_DIR/runs/multilinguality_model/logs"

while true; do
  echo "=== Eval loop pass: $(date) ===" >> "$LOGS/eval_loop.log"
  MERGED_ROOT="multilingual/models/merged/v1_v20/" \
  RESULTS_ROOT="multilingual/eval/results/merged/v1_v20/" \
  bash "$REPO_DIR/code/multilinguality_model/eval/bash/eval_merged.sh" \
    >> "$LOGS/eval_loop.log" 2>&1 || true
  echo "=== Eval loop done, sleeping 60s: $(date) ===" >> "$LOGS/eval_loop.log"
  sleep 60
done