#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root

MODEL="$PROJECT_DIR/models/sft_grpo_dpo"
BENCHMARKS="$PROJECT_DIR/data/preprocessing/benchmarks"
OUTPUT_DIR="$PROJECT_DIR/results/sft_grpo_dpo/benchmarks"

mkdir -p "$OUTPUT_DIR"

for dir in "$BENCHMARKS"/*/; do
    subject=$(basename "$dir")
    case "$subject" in
        cais_mmlu|MMLU-pro) continue ;;
    esac

    jsonl=$(ls "$dir"*.jsonl 2>/dev/null | head -1)
    if [ -z "$jsonl" ]; then
        echo "Skipping $subject: no .jsonl file found"
        continue
    fi

    echo "=== $subject ==="
    python "$GK_DIR/scripts/evaluate_training.py" \
        --checkpoints-dir "$MODEL" \
        --validation-file "$jsonl" \
        --output-dir "$OUTPUT_DIR/$subject" \
        --skip-wandb
done

for dir in "$BENCHMARKS/cais_mmlu"/*/; do
    subject=$(basename "$dir")

    jsonl=$(ls "$dir"*.jsonl 2>/dev/null | head -1)
    if [ -z "$jsonl" ]; then
        echo "Skipping cais_mmlu/$subject: no .jsonl file found"
        continue
    fi

    echo "=== cais_mmlu/$subject ==="
    python "$GK_DIR/scripts/evaluate_training.py" \
        --checkpoints-dir "$MODEL" \
        --validation-file "$jsonl" \
        --output-dir "$OUTPUT_DIR/$subject" \
        --skip-wandb
done

python "$GK_DIR/scripts/summarize_benchmarks.py" --benchmarks-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/benchmark_summary.json"