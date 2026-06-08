#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/code:${PYTHONPATH:-}"

CHAT_TEMPLATE="code/multilinguality_model/assets/chat_template_multilingual-nonthinking.jinja"

FULL_BENCHMARK_FILE="data/multilinguality/benchmarks/benchmark_all.jsonl"
PROVIDED_FILE="validation_samples/multilingual.jsonl"
MMLU_PROX_FILE="data/multilinguality/benchmarks/mmlu_prox/mmlu_prox_hard_1000.jsonl"

RESULTS_ROOT="multilingual/eval/results/sft"

MODEL_SPECS=(
  # "base-hf|Qwen/Qwen3-1.7B"
  # "base|$REPO_DIR/checkpoints/multilinguality_model/Qwen3-1.7B-baseline"
  # "bilko|cs-552-2026-bilko/multilingual_model"
  # "flab|cs-552-2026-flab/multilingual_model"
  # "qwenlifegivesyoulemons|cs-552-2026-qwenlifegivesyoulemons/multilingual_model"

  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v1"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v16_nonthinking_1e-4_warmup0p1"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v17_10k_nonthinking_1e-4_warmup0p15"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v17_20k_nonthinking_5e-5_warmup0p15"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v18_nonthinking_5e-5_warmup0p15"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v19_15k_nonthinking_5e-5_warmup0p15"
  # "$REPO_DIR/checkpoints/multilinguality_model/sft/sft_multilingual_v19_15k_nonthinking_5e-5_warmup0p15"
  # "$REPO_DIR/checkpoints/multilinguality_model/exports/sft_multilingual_v19_15k_augmented_nonthinking_5e-5_warmup0p15"
  # "$REPO_DIR/checkpoints/multilinguality_model/sft/sft_multilingual_v19_20k_augmented_nonthinking_5e-5_warmup0p15"
  # "$REPO_DIR/checkpoints/multilinguality_model/sft/sft_multilingual_v19_20k_nonthinking_5e-5_warmup0p15"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/v20-chinese"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/v20-hindi"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/v20-russian"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/v20-italian"
  "$REPO_DIR/checkpoints/multilinguality_model/exports/v20-spanish"
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
    if [[ ! -f "$path" ]]; then
      echo "[ERROR] Missing file: $path"
      missing=1
    fi
  done

  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

suite_primary_result_file() {
  local suite_name="$1"

  if [[ "$suite_name" == "validation_set_mnlp" ]]; then
    echo "provided_results.jsonl"
  else
    echo "val_results.jsonl"
  fi
}

suite_is_complete() {
  local suite_name="$1"
  local suite_dir="$2"
  local primary_file
  primary_file="$(suite_primary_result_file "$suite_name")"

  if [[ ! -d "$suite_dir" ]]; then
    return 1
  fi

  if [[ ! -s "$suite_dir/$primary_file" ]]; then
    return 1
  fi

  if [[ ! -s "$suite_dir/all_results.jsonl" ]]; then
    return 1
  fi

  if [[ ! -s "$suite_dir/summary.txt" ]]; then
    return 1
  fi

  if [[ ! -s "$suite_dir/eval.log" ]]; then
    return 1
  fi

  if grep -qi "\[ERROR\]\|Traceback\|RuntimeError\|OutOfMemoryError\|CUDA out of memory\|Engine core initialization failed" "$suite_dir/eval.log"; then
    return 1
  fi

  return 0
}

model_is_complete() {
  local model_results_dir="$1"

  suite_is_complete "full" "$model_results_dir/full" &&
  suite_is_complete "validation_set_mnlp" "$model_results_dir/validation_set_mnlp" &&
  suite_is_complete "mmlu_prox_hard" "$model_results_dir/mmlu_prox_hard"
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

  local extra_args=()

  if [[ "$suite_name" == "full" ]]; then
    extra_args=(--val_file "$FULL_BENCHMARK_FILE")
  elif [[ "$suite_name" == "validation_set_mnlp" ]]; then
    extra_args=(--provided_only --provided_file "$PROVIDED_FILE")
  elif [[ "$suite_name" == "mmlu_prox_hard" ]]; then
    extra_args=(--val_file "$MMLU_PROX_FILE")
  else
    echo "[ERROR] Unknown suite: $suite_name"
    return 1
  fi

  set +e
  python -m multilinguality_model.eval.run_eval_vllm \
    --model_name "$model_path" \
    "${extra_args[@]}" \
    --output_dir "$out_dir" \
    --chat_template "$CHAT_TEMPLATE" \
    --mcq_max_new_tokens 4096 \
    --math_max_new_tokens 4096 \
    --max_model_len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --overwrite \
    2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[ERROR] Failed: $model_name / $suite_name with exit code $status" | tee -a "$log_file"
    return "$status"
  fi

  if ! suite_is_complete "$suite_name" "$out_dir"; then
    echo "[ERROR] Run ended but expected result files are missing/incomplete: $model_name / $suite_name" | tee -a "$log_file"
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
echo "Decoding:        greedy/no-sampling by default; --do_sample is not passed"
echo "Results root:    $RESULTS_ROOT"
echo "Started:         $(date)"
echo "============================================================"

check_files

FAILED=()

for SPEC in "${MODEL_SPECS[@]}"; do
  parse_spec "$SPEC"

  NAME="$PARSED_NAME"
  MODEL_PATH="$PARSED_MODEL_PATH"
  MODEL_RESULTS_DIR="$RESULTS_ROOT/$NAME"

  if [[ "$MODEL_PATH" == /* ]] && [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing model directory: $MODEL_PATH"
    FAILED+=("$NAME : missing model dir")
    continue
  fi

  if model_is_complete "$MODEL_RESULTS_DIR"; then
    echo
    echo "============================================================"
    echo "Skipping already completed model: $NAME"
    echo "Results dir: $MODEL_RESULTS_DIR"
    echo "============================================================"
    continue
  fi

  if suite_is_complete "full" "$MODEL_RESULTS_DIR/full"; then
    echo "[SKIP] $NAME / full already complete"
  else
    if ! run_eval_suite "$NAME" "$MODEL_PATH" "full" "$MODEL_RESULTS_DIR/full"; then
      FAILED+=("$NAME/full")
      continue
    fi
  fi

  if suite_is_complete "validation_set_mnlp" "$MODEL_RESULTS_DIR/validation_set_mnlp"; then
    echo "[SKIP] $NAME / validation_set_mnlp already complete"
  else
    if ! run_eval_suite "$NAME" "$MODEL_PATH" "validation_set_mnlp" "$MODEL_RESULTS_DIR/validation_set_mnlp"; then
      FAILED+=("$NAME/validation_set_mnlp")
      continue
    fi
  fi

  if suite_is_complete "mmlu_prox_hard" "$MODEL_RESULTS_DIR/mmlu_prox_hard"; then
    echo "[SKIP] $NAME / mmlu_prox_hard already complete"
  else
    if ! run_eval_suite "$NAME" "$MODEL_PATH" "mmlu_prox_hard" "$MODEL_RESULTS_DIR/mmlu_prox_hard"; then
      FAILED+=("$NAME/mmlu_prox_hard")
      continue
    fi
  fi
done

echo
echo "============================================================"
echo "Batch eval finished: $(date)"
echo "============================================================"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed runs:"
  for item in "${FAILED[@]}"; do
    echo "  - $item"
  done
  exit 1
else
  echo "All eval suites completed successfully."
fi