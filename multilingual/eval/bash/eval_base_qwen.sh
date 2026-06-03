#!/bin/bash
set -euo pipefail

cd /scratch/workspace/emma/standard-project-m2-mystery-machine

MODEL_NAME="Qwen/Qwen3-1.7B"
CHAT_TEMPLATE="multilingual/assets/chat_template_multilingual-v2.jinja"
VAL_FILE="multilingual/data/benchmarks/benchmark_all.jsonl"
PROVIDED_FILE="validation_samples/multilingual.jsonl"

OUT_DIR="multilingual/eval/results/base/full"
LOG_FILE="$OUT_DIR/eval.log"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Base Qwen3-1.7B vLLM eval on full benchmark + validation samples"
echo "Model:         $MODEL_NAME"
echo "Val file:      $VAL_FILE"
echo "Provided file: $PROVIDED_FILE"
echo "Template:      $CHAT_TEMPLATE"
echo "Output dir:    $OUT_DIR"
echo "Started:       $(date)"
echo "============================================================"

python -m multilingual.eval.run_eval_vllm \
  --model_name "$MODEL_NAME" \
  --val_file "$VAL_FILE" \
  --eval_provided \
  --provided_file "$PROVIDED_FILE" \
  --output_dir "$OUT_DIR" \
  --chat_template "$CHAT_TEMPLATE" \
  --mcq_max_new_tokens 4096 \
  --math_max_new_tokens 4096 \
  --max_model_len 8192 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.90 \
  --overwrite \
  2>&1 | tee "$LOG_FILE"