#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/code:${PYTHONPATH:-}"

CHAT_TEMPLATE="code/multilinguality_model/assets/chat_template_multilingual-thinking.jinja"

FULL_BENCHMARK_FILE="data/multilinguality/benchmarks/benchmark_all.jsonl"
PROVIDED_FILE="validation_samples/multilingual.jsonl"
MMLU_PROX_FILE="data/multilinguality/benchmarks/mmlu_prox/mmlu_prox_hard_1000.jsonl"

RESULTS_ROOT="multilingual/eval/results/sft-do-sample/"

MODEL_SPECS=(
  "bilko|cs-552-2026-bilko/multilingual_model"
  "flab|cs-552-2026-flab/multilingual_model"
  "qwenlifegivesyoulemons|cs-552-2026-qwenlifegivesyoulemons/multilingual_model"

  # "$REPO_DIR/checkpoints/multilinguality_model/exports/debug_v0"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v1"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v8_distilled"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v9_distilled_shorter"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v10_distilled"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v10_distilled_short"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v11_distilled_only"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v11_distilled_only_short"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v11_distilled_only_warmup0p1"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v12_distilled"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v12_distilled_only_1e-4_warmup0p1_v2_template"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v13_distilled_1e-4_warmup0p1"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v14_distilled_1e-4_warmup0p1"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v14_distilled_2e-4_warmup0p05_short"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v15_distilled_5e-5_warmup0p1"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v16_nonthinking_1e-4_warmup0p1"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v17_10k_nonthinking_1e-4_warmup0p15"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v17_20k_nonthinking_5e-5_warmup0p15"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v18_nonthinking_5e-5_warmup0p15"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v19_15k_nonthinking_5e-5_warmup0p15"
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

  local val_file=""
  if [ "$suite_name" = "full" ]; then
    val_file="$FULL_BENCHMARK_FILE"
  elif [ "$suite_name" = "validation_set_mnlp" ]; then
    val_file=""  # provided_only
  elif [ "$suite_name" = "mmlu_prox_hard" ]; then
    val_file="$MMLU_PROX_FILE"
  else
    echo "[ERROR] Unknown suite: $suite_name"
    return 1
  fi

  local extra_args=""
  if [ "$suite_name" = "validation_set_mnlp" ]; then
    extra_args="--provided_only --provided_file $PROVIDED_FILE"
  else
    extra_args="--val_file $val_file"
  fi

  # shellcheck disable=SC2086
  python -m multilinguality_model.eval.run_eval_vllm \
    --model_name "$model_path" \
    $extra_args \
    --output_dir "$out_dir" \
    --chat_template "$CHAT_TEMPLATE" \
    --mcq_max_new_tokens 4096 \
    --math_max_new_tokens 4096 \
    --max_model_len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --do_sample \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --overwrite \
    2>&1 | tee "$log_file"

  echo "Finished OK: $model_name / $suite_name at $(date)" | tee -a "$log_file"
}

echo "============================================================"
echo "Batch vLLM eval (do_sample): full + validation_set_mnlp + MMLU-ProX hard"
echo "Full benchmark:  $FULL_BENCHMARK_FILE"
echo "Validation set:  $PROVIDED_FILE"
echo "MMLU-ProX file:  $MMLU_PROX_FILE"
echo "Template:        $CHAT_TEMPLATE"
echo "Sampling:        temperature=0.6  top_p=0.95  top_k=20"
echo "Results root:    $RESULTS_ROOT"
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

  if ! run_eval_suite "$NAME" "$MODEL_PATH" "full"               "$RESULTS_ROOT/$NAME/full";               then FAILED+=("$NAME/full");               continue; fi
  if ! run_eval_suite "$NAME" "$MODEL_PATH" "validation_set_mnlp" "$RESULTS_ROOT/$NAME/validation_set_mnlp"; then FAILED+=("$NAME/validation_set_mnlp"); continue; fi
  if ! run_eval_suite "$NAME" "$MODEL_PATH" "mmlu_prox_hard"     "$RESULTS_ROOT/$NAME/mmlu_prox_hard";     then FAILED+=("$NAME/mmlu_prox_hard");     continue; fi

done

echo
echo "============================================================"
echo "do_sample batch eval finished: $(date)"
echo "============================================================"

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Failed runs:"
  for item in "${FAILED[@]}"; do
    echo "  - $item"
  done
  exit 1
else
  echo "All do_sample eval suites completed successfully."
fi