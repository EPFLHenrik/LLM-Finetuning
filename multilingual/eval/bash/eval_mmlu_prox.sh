#!/bin/bash
set -euo pipefail

cd /scratch/workspace/emma/standard-project-m2-mystery-machine

CHAT_TEMPLATE="multilingual/assets/chat_template_multilingual-v2.jinja"
MMLU_PROX_FILE="multilingual/data/benchmarks/mmlu_prox/mmlu_prox_hard_1000.jsonl"

RESULTS_ROOT="multilingual/eval/results"

# Format:
#   "display_name|model_path_or_hf_id"
MODEL_SPECS=(
  # "base|Qwen/Qwen3-1.7B"
  "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/debug_v0"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v1"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v8_distilled"
#  "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v9_distilled_shorter"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v10_distilled"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v10_distilled_short"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v11_distilled_only"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v11_distilled_only_short"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v11_distilled_only_warmup0p1"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v12_distilled"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v12_distilled_only_1e-4_warmup0p1_v2_template"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v13_distilled_1e-4_warmup0p1"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v14_distilled_1e-4_warmup0p1"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v14_distilled_2e-4_warmup0p05_short"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v15_distilled_5e-5_warmup0p1"
 "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v16_nonthinking_1e-4_warmup0p1"
"/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v17_10k_nonthinking_1e-4_warmup0p15"
"/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v17_20k_nonthinking_5e-5_warmup0p15"
)

run_mmlu_prox_eval() {
  local model_name="$1"
  local model_path="$2"
  local out_dir="$3"
  local log_file="$out_dir/eval.log"

  mkdir -p "$out_dir"

  echo
  echo "============================================================"
  echo "Evaluating model: $model_name"
  echo "Suite:            mmlu_prox_hard"
  echo "Model path/name:  $model_path"
  echo "Benchmark file:   $MMLU_PROX_FILE"
  echo "Output dir:       $out_dir"
  echo "Started:          $(date)"
  echo "============================================================"

  python -m multilingual.eval.run_eval_vllm \
    --model_name "$model_path" \
    --val_file "$MMLU_PROX_FILE" \
    --output_dir "$out_dir" \
    --chat_template "$CHAT_TEMPLATE" \
    --mcq_max_new_tokens 4096 \
    --math_max_new_tokens 4096 \
    --max_model_len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --overwrite \
    2>&1 | tee "$log_file"

  echo "Finished OK: $model_name / mmlu_prox_hard at $(date)" | tee -a "$log_file"
}

echo "============================================================"
echo "Batch vLLM eval: selected models on MMLU-ProX hard"
echo "MMLU-ProX file:  $MMLU_PROX_FILE"
echo "Template:        $CHAT_TEMPLATE"
echo "Started:         $(date)"
echo "============================================================"

if [ ! -f "$MMLU_PROX_FILE" ]; then
  echo "[ERROR] Missing benchmark file: $MMLU_PROX_FILE"
  exit 1
fi

FAILED=()

for SPEC in "${MODEL_SPECS[@]}"; do
  if [[ "$SPEC" == *"|"* ]]; then
    NAME="${SPEC%%|*}"
    MODEL_PATH="${SPEC#*|}"
  else
    MODEL_PATH="$SPEC"
    NAME="$(basename "$MODEL_PATH")"
    NAME="${NAME#sft_multilingual_}"
  fi

  # For local exported models, check the directory exists.
  # For HF model IDs like Qwen/Qwen3-1.7B, skip this check.
  if [[ "$MODEL_PATH" == /* ]] && [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Missing model directory: $MODEL_PATH"
    FAILED+=("$NAME : missing model dir")
    continue
  fi

  OUT_DIR="$RESULTS_ROOT/$NAME/mmlu_prox_hard"

  if ! run_mmlu_prox_eval "$NAME" "$MODEL_PATH" "$OUT_DIR"; then
    echo "[ERROR] MMLU-ProX hard eval failed: $NAME"
    FAILED+=("$NAME/mmlu_prox_hard")
    continue
  fi
done

echo
echo "============================================================"
echo "MMLU-ProX batch eval finished: $(date)"
echo "============================================================"

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Failed runs:"
  for item in "${FAILED[@]}"; do
    echo "  - $item"
  done
  exit 1
else
  echo "All MMLU-ProX evals completed successfully."
fi