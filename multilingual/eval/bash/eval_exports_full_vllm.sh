#!/bin/bash
set -euo pipefail

cd /scratch/workspace/emma/standard-project-m2-mystery-machine

CHAT_TEMPLATE="multilingual/assets/chat_template_multilingual-v2.jinja"

FULL_BENCHMARK_FILE="multilingual/data/benchmarks/benchmark_all.jsonl"
PROVIDED_FILE="validation_samples/multilingual.jsonl"
MMLU_PROX_FILE="multilingual/data/benchmarks/mmlu_prox/mmlu_prox_hard_1000.jsonl"

RESULTS_ROOT="multilingual/eval/results"

# Format can be either:
#   "display_name|model_path_or_hf_id"
# or:
#   "/absolute/path/to/exported/model"
#
# Plain exported paths are automatically converted:
#   .../sft_multilingual_v1 -> v1
#   .../debug_v0           -> debug_v0
MODEL_SPECS=(
  # "base|Qwen/Qwen3-1.7B"

  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/debug_v0"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v1"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v8_distilled"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v9_distilled_shorter"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v10_distilled"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v10_distilled_short"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v11_distilled_only"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v11_distilled_only_short"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v11_distilled_only_warmup0p1"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v12_distilled"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v12_distilled_only_1e-4_warmup0p1_v2_template"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v13_distilled_1e-4_warmup0p1"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v14_distilled_1e-4_warmup0p1"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v14_distilled_2e-4_warmup0p05_short"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v15_distilled_5e-5_warmup0p1"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v16_nonthinking_1e-4_warmup0p1"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v17_10k_nonthinking_1e-4_warmup0p15"
  # "/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v17_20k_nonthinking_5e-5_warmup0p15"
"/scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/results/exports/sft_multilingual_v18_nonthinking_5e-5_warmup0p15"
)

parse_spec() {
  local spec="$1"

  if [[ "$spec" == *"|"* ]]; then
    PARSED_NAME="${spec%%|*}"
    PARSED_MODEL_PATH="${spec#*|}"
  else
    PARSED_MODEL_PATH="$spec"
    PARSED_NAME="$(basename "$PARSED_MODEL_PATH")"
    PARSED_NAME="${PARSED_NAME#sft_multilingual_}"
  fi
}

check_files() {
  local missing=0

  for path in "$FULL_BENCHMARK_FILE" "$PROVIDED_FILE" "$MMLU_PROX_FILE" "$CHAT_TEMPLATE"; do
    if [ ! -f "$path" ]; then
      echo "[ERROR] Missing file: $path"
      missing=1
    fi
  done

  if [ "$missing" -ne 0 ]; then
    exit 1
  fi
}

run_eval_suite() {
  local model_name="$1"
  local model_path="$2"
  local suite_name="$3"
  local out_dir="$4"
  local log_file="$out_dir/eval.log"

  mkdir -p "$out_dir"

  echo
  echo "============================================================"
  echo "Evaluating model: $model_name"
  echo "Suite:            $suite_name"
  echo "Model path/name:  $model_path"
  echo "Output dir:       $out_dir"
  echo "Started:          $(date)"
  echo "============================================================"

  if [ "$suite_name" = "full" ]; then
    python -m multilingual.eval.run_eval_vllm \
      --model_name "$model_path" \
      --val_file "$FULL_BENCHMARK_FILE" \
      --output_dir "$out_dir" \
      --chat_template "$CHAT_TEMPLATE" \
      --mcq_max_new_tokens 4096 \
      --math_max_new_tokens 4096 \
      --max_model_len 8192 \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.90 \
      --overwrite \
      2>&1 | tee "$log_file"

  elif [ "$suite_name" = "validation_set_mnlp" ]; then
    python -m multilingual.eval.run_eval_vllm \
      --model_name "$model_path" \
      --provided_only \
      --provided_file "$PROVIDED_FILE" \
      --output_dir "$out_dir" \
      --chat_template "$CHAT_TEMPLATE" \
      --mcq_max_new_tokens 4096 \
      --math_max_new_tokens 4096 \
      --max_model_len 8192 \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.90 \
      --overwrite \
      2>&1 | tee "$log_file"

  elif [ "$suite_name" = "mmlu_prox_hard" ]; then
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

  else
    echo "[ERROR] Unknown suite: $suite_name"
    return 1
  fi

  echo "Finished OK: $model_name / $suite_name at $(date)" | tee -a "$log_file"
}

echo "============================================================"
echo "Batch vLLM eval: full + validation_set_mnlp + MMLU-ProX hard"
echo "Full benchmark:  $FULL_BENCHMARK_FILE"
echo "Validation set:  $PROVIDED_FILE"
echo "MMLU-ProX file:  $MMLU_PROX_FILE"
echo "Template:        $CHAT_TEMPLATE"
echo "Started:         $(date)"
echo "============================================================"

check_files

FAILED=()

for SPEC in "${MODEL_SPECS[@]}"; do
  parse_spec "$SPEC"

  NAME="$PARSED_NAME"
  MODEL_PATH="$PARSED_MODEL_PATH"

  if [[ "$MODEL_PATH" == /* ]] && [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Missing model directory: $MODEL_PATH"
    FAILED+=("$NAME : missing model dir")
    continue
  fi

  FULL_OUT_DIR="$RESULTS_ROOT/$NAME/full"
  VALIDATION_OUT_DIR="$RESULTS_ROOT/$NAME/validation_set_mnlp"
  MMLU_PROX_OUT_DIR="$RESULTS_ROOT/$NAME/mmlu_prox_hard"

  if ! run_eval_suite "$NAME" "$MODEL_PATH" "full" "$FULL_OUT_DIR"; then
    echo "[ERROR] Full benchmark eval failed: $NAME"
    FAILED+=("$NAME/full")
    continue
  fi

  if ! run_eval_suite "$NAME" "$MODEL_PATH" "validation_set_mnlp" "$VALIDATION_OUT_DIR"; then
    echo "[ERROR] Validation-set eval failed: $NAME"
    FAILED+=("$NAME/validation_set_mnlp")
    continue
  fi

  if ! run_eval_suite "$NAME" "$MODEL_PATH" "mmlu_prox_hard" "$MMLU_PROX_OUT_DIR"; then
    echo "[ERROR] MMLU-ProX eval failed: $NAME"
    FAILED+=("$NAME/mmlu_prox_hard")
    continue
  fi
done

echo
echo "============================================================"
echo "Batch eval finished: $(date)"
echo "============================================================"

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Failed runs:"
  for item in "${FAILED[@]}"; do
    echo "  - $item"
  done
  exit 1
else
  echo "All eval suites completed successfully."
fi