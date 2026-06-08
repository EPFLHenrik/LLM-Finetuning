#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "${REPO_DIR}"
source code/multilinguality_model/setup_env.sh

MODEL_DIR="checkpoints/multilinguality_model/best_model"
CHAT_TEMPLATE="code/multilinguality_model/assets/chat_template_multilingual-nonthinking.jinja"

BENCHMARK_FILE="data/multilinguality/benchmarks/benchmark_all.jsonl"
MMLU_PROX_FILE="data/multilinguality/benchmarks/mmlu_prox/mmlu_prox_hard_1000.jsonl"

RESULT_ROOT="runs/benchmark/results/multilinguality_model/best_model"
FULL_EVAL_DIR="${RESULT_ROOT}/benchmark_all"
MMLU_PROX_EVAL_DIR="${RESULT_ROOT}/mmlu_prox_hard"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${BENCHMARK_FILE}" ]]; then
  echo "Missing benchmark file: ${BENCHMARK_FILE}" >&2
  exit 1
fi

mkdir -p "${FULL_EVAL_DIR}" "${MMLU_PROX_EVAL_DIR}"

echo "Evaluating benchmark_all"

python -m multilinguality_model.eval.run_eval_vllm \
  --model_name "${MODEL_DIR}" \
  --val_file "${BENCHMARK_FILE}" \
  --output_dir "${FULL_EVAL_DIR}" \
  --chat_template "${CHAT_TEMPLATE}" \
  --system_prompt non_thinking \
  --mcq_max_new_tokens 4096 \
  --math_max_new_tokens 4096 \
  --max_model_len 8192 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.90 \
  --overwrite \
  2>&1 | tee "${FULL_EVAL_DIR}/eval.log"

if [[ -f "${MMLU_PROX_FILE}" ]]; then
  echo "Evaluating MMLU-ProX hard"

  python -m multilinguality_model.eval.run_eval_vllm \
    --model_name "${MODEL_DIR}" \
    --val_file "${MMLU_PROX_FILE}" \
    --output_dir "${MMLU_PROX_EVAL_DIR}" \
    --chat_template "${CHAT_TEMPLATE}" \
  --system_prompt non_thinking \
    --mcq_max_new_tokens 4096 \
    --math_max_new_tokens 4096 \
    --max_model_len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --overwrite \
    2>&1 | tee "${MMLU_PROX_EVAL_DIR}/eval.log"
else
  echo "MMLU-ProX file not found, skipping: ${MMLU_PROX_FILE}"
fi

echo "Aggregating results"

python -m multilinguality_model.eval.aggregate_results \
  --result_root "runs/benchmark/results/multilinguality_model"

echo "Evaluation finished"
echo "Results root: ${RESULT_ROOT}"
