#!/usr/bin/env bash
set -euo pipefail

MODEL="models/qwen3_thinking"
BENCHMARKS="data/preprocessing/benchmarks"
OUTPUT_DIR="results/qwen3_thinking/benchmarks"

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

    if [ -f "$OUTPUT_DIR/$subject/evaluation_summary.json" ]; then
        echo "Skipping $subject: results already exist"
        continue
    fi

    echo "=== $subject ==="
    python scripts/evaluate_training.py \
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

    if [ -f "$OUTPUT_DIR/$subject/evaluation_summary.json" ]; then
        echo "Skipping cais_mmlu/$subject: results already exist"
        continue
    fi

    echo "=== cais_mmlu/$subject ==="
    python scripts/evaluate_training.py \
        --checkpoints-dir "$MODEL" \
        --validation-file "$jsonl" \
        --output-dir "$OUTPUT_DIR/$subject" \
        --skip-wandb
done

python scripts/summarize_benchmarks.py --benchmarks-dir results/qwen3_thinking/benchmarks --output results/qwen3_thinking/benchmarks/benchmark_summary.json