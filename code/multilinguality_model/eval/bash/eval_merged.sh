#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/code:${PYTHONPATH:-}"

MERGED_ROOT="${MERGED_ROOT:-multilingual/models/merged/v20/}"
RESULTS_ROOT="${RESULTS_ROOT:-multilingual/eval/results/merged/v20/}"

CHAT_TEMPLATE="code/multilinguality_model/assets/chat_template_multilingual-nonthinking.jinja"

FULL_BENCHMARK_FILE="data/multilinguality/benchmarks/benchmark_all.jsonl"
PROVIDED_FILE="validation_samples/multilingual.jsonl"
MMLU_PROX_FILE="data/multilinguality/benchmarks/mmlu_prox/mmlu_prox_hard_1000.jsonl"

check_gpu() {
  echo "Checking GPU visibility..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "[ERROR] nvidia-smi not found. Run this from a GPU node."
    exit 1
  fi

  python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] No CUDA device visible.")
PY
}

check_files() {
  local missing=0

  for path in "$CHAT_TEMPLATE" "$FULL_BENCHMARK_FILE" "$PROVIDED_FILE" "$MMLU_PROX_FILE"; do
    if [[ ! -f "$path" ]]; then
      echo "[ERROR] Missing file: $path"
      missing=1
    fi
  done

  if [[ ! -d "$MERGED_ROOT" ]]; then
    echo "[ERROR] Missing merged model root: $MERGED_ROOT"
    missing=1
  fi

  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

discover_models() {
  mapfile -t MODEL_DIRS < <(find "$MERGED_ROOT" -mindepth 1 -maxdepth 2 -type d | grep -v '/step[0-9]' | grep -v '/ord[0-9]' | sort)

  if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] No subfolders found under $MERGED_ROOT"
    exit 1
  fi

  echo ""
  echo "Discovered merged model folders:"
  for model_dir in "${MODEL_DIRS[@]}"; do
    if [[ -f "$model_dir/config.json" ]]; then
      echo "  OK   $model_dir"
    else
      echo "  SKIP $model_dir  missing config.json"
    fi
  done
}

run_eval_suite() {
  local model_name="$1"
  local model_path="$2"
  local suite_name="$3"
  local out_dir="$4"
  local log_file="$out_dir/eval.log"

  mkdir -p "$out_dir"

  echo ""
  echo "============================================================"
  echo "Evaluating merged model: $model_name"
  echo "Suite:                  $suite_name"
  echo "Model path:             $model_path"
  echo "Output dir:             $out_dir"
  echo "Started:                $(date)"
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

    echo "Finished OK: $model_name / $suite_name at $(date)" | tee -a "$log_file"
}

echo "============================================================"
echo "Merged model vLLM evaluation"
echo "Merged root:      $MERGED_ROOT"
echo "Results root:     $RESULTS_ROOT"
echo "Full benchmark:   $FULL_BENCHMARK_FILE"
echo "Validation set:   $PROVIDED_FILE"
echo "MMLU-ProX file:   $MMLU_PROX_FILE"
echo "Chat template:    $CHAT_TEMPLATE"
echo "Started:          $(date)"
echo "============================================================"

check_gpu
check_files
discover_models

FAILED=()

suite_is_complete() {
  local suite_dir="$1"

  if [[ ! -f "$suite_dir/summary.txt" ]]; then
    return 1
  fi

  if [[ ! -f "$suite_dir/eval.log" ]]; then
    return 1
  fi

  if grep -qi "\[ERROR\]\|Traceback\|RuntimeError\|OutOfMemoryError" "$suite_dir/eval.log"; then
    return 1
  fi

  return 0
}

model_is_complete() {
  local model_results_dir="$1"

  suite_is_complete "$model_results_dir/full" &&
  suite_is_complete "$model_results_dir/validation_set_mnlp" &&
  suite_is_complete "$model_results_dir/mmlu_prox_hard"
}


for MODEL_PATH in "${MODEL_DIRS[@]}"; do
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    continue
  fi

  MODEL_NAME="$(basename "$MODEL_PATH")"
  MODEL_RESULTS_DIR="$RESULTS_ROOT/$MODEL_NAME"

  if model_is_complete "$MODEL_RESULTS_DIR"; then
    echo ""
    echo "============================================================"
    echo "Skipping already completed model: $MODEL_NAME"
    echo "Results dir: $MODEL_RESULTS_DIR"
    echo "============================================================"
    continue
  fi

  if suite_is_complete "$MODEL_RESULTS_DIR/full"; then
    echo "[SKIP] $MODEL_NAME / full already complete"
  else
    if ! run_eval_suite "$MODEL_NAME" "$MODEL_PATH" "full" "$MODEL_RESULTS_DIR/full"; then
      FAILED+=("$MODEL_NAME/full")
      continue
    fi
  fi

  if suite_is_complete "$MODEL_RESULTS_DIR/validation_set_mnlp"; then
    echo "[SKIP] $MODEL_NAME / validation_set_mnlp already complete"
  else
    if ! run_eval_suite "$MODEL_NAME" "$MODEL_PATH" "validation_set_mnlp" "$MODEL_RESULTS_DIR/validation_set_mnlp"; then
      FAILED+=("$MODEL_NAME/validation_set_mnlp")
      continue
    fi
  fi

  if suite_is_complete "$MODEL_RESULTS_DIR/mmlu_prox_hard"; then
    echo "[SKIP] $MODEL_NAME / mmlu_prox_hard already complete"
  else
    if ! run_eval_suite "$MODEL_NAME" "$MODEL_PATH" "mmlu_prox_hard" "$MODEL_RESULTS_DIR/mmlu_prox_hard"; then
      FAILED+=("$MODEL_NAME/mmlu_prox_hard")
      continue
    fi
  fi
done

echo ""
echo "============================================================"
echo "Merged model eval finished: $(date)"
echo "============================================================"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed runs:"
  for item in "${FAILED[@]}"; do
    echo "  - $item"
  done
  exit 1
else
  echo "All merged model eval suites completed successfully."
fi
