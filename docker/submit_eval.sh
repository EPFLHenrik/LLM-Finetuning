#!/bin/bash
# Submit an overnight evaluation job.
# Evaluates base Qwen + v5/v6/v8/v9 checkpoints on INCLUDE val + 10 provided examples.
# Runs sequentially to avoid OOM. Results saved per model in multilingual/results/eval/.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="anastass"
GROUP="g02"
REPO_DIR="/scratch/workspace/emma/standard-project-m2-mystery-machine"

VAL_FILE="${REPO_DIR}/multilingual/data/sft/v9/mixed/include_val.jsonl"
EVAL_DIR="${REPO_DIR}/multilingual/results/eval"
LATEST_CHECKPOINT=700

V5_PATH="${REPO_DIR}/multilingual/results/checkpoints/sft_multilingual_v5-less-data"
V6_PATH="${REPO_DIR}/multilingual/results/checkpoints/sft_multilingual_v6-ru-zh-heavy"
V8_PATH="${REPO_DIR}/multilingual/results/checkpoints/sft_multilingual_v8_distilled/best_adapter"
V9S_PATH="${REPO_DIR}/multilingual/results/checkpoints/sft_multilingual_v9_distilled_shorter/checkpoint-${LATEST_CHECKPOINT}"
V9_PATH="${REPO_DIR}/multilingual/results/checkpoints/sft_multilingual_v9_distilled/checkpoint-${LATEST_CHECKPOINT}"
# ==============================================

if [[ -z "${GASPAR}" ]]; then
    echo "ERROR: set GASPAR." >&2; exit 1
fi
if [[ -z "${GROUP}" ]]; then
    echo "ERROR: set GROUP." >&2; exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=..." >&2; exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-eval-overnight}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

echo ">>> Submitting ${JOB_NAME}"
echo ">>> Val file:  ${VAL_FILE}"
echo ">>> Eval dir:  ${EVAL_DIR}"

MSYS_NO_PATHCONV=1 runai training submit "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu-devices-request "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd ${REPO_DIR} && \
    echo '[eval] Starting evaluation run at \$(date)' && \

    echo '[eval] 1/6 Base Qwen3-1.7B' && \
    python -m multilingual.eval.run_eval \
      --model_name Qwen/Qwen3-1.7B \
      --no_custom_template \
      --val_file ${VAL_FILE} \
      --eval_provided \
      --output_dir ${EVAL_DIR}/base_qwen \
      --overwrite && \

    echo '[eval] 2/6 V5' && \
    python -m multilingual.eval.run_eval \
      --model_name ${V5_PATH} \
      --val_file ${VAL_FILE} \
      --eval_provided \
      --output_dir ${EVAL_DIR}/v5 \
      --overwrite && \

    echo '[eval] 3/6 V6' && \
    python -m multilingual.eval.run_eval \
      --model_name ${V6_PATH} \
      --val_file ${VAL_FILE} \
      --eval_provided \
      --output_dir ${EVAL_DIR}/v6 \
      --overwrite && \

    echo '[eval] 4/6 V8' && \
    python -m multilingual.eval.run_eval \
      --model_name ${V8_PATH} \
      --val_file ${VAL_FILE} \
      --eval_provided \
      --output_dir ${EVAL_DIR}/v8 \
      --overwrite && \

    echo '[eval] 5/6 V9 shorter checkpoint-${LATEST_CHECKPOINT}' && \
    python -m multilingual.eval.run_eval \
      --model_name ${V9S_PATH} \
      --val_file ${VAL_FILE} \
      --eval_provided \
      --output_dir ${EVAL_DIR}/v9_shorter_ckpt${LATEST_CHECKPOINT} \
      --overwrite && \

    echo '[eval] 6/6 V9 checkpoint-${LATEST_CHECKPOINT}' && \
    python -m multilingual.eval.run_eval \
      --model_name ${V9_PATH} \
      --val_file ${VAL_FILE} \
      --eval_provided \
      --output_dir ${EVAL_DIR}/v9_ckpt${LATEST_CHECKPOINT} \
      --overwrite && \

    echo '[eval] All done at \$(date)'"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:
  runai workload describe ${JOB_NAME} -p ${PROJECT}

Stream logs:
  runai workload logs ${JOB_NAME} -p ${PROJECT}

If above doesn't work:
  runai logs -f ${JOB_NAME} -p ${PROJECT}

Shell in pod:
  runai bash ${JOB_NAME} -p ${PROJECT}

Stop the job:
  runai workload delete ${JOB_NAME} -p ${PROJECT}

Results will be saved in:
  ${EVAL_DIR}/base_qwen/summary.csv
  ${EVAL_DIR}/v5/summary.csv
  ${EVAL_DIR}/v6/summary.csv
  ${EVAL_DIR}/v8/summary.csv
  ${EVAL_DIR}/v9_shorter_ckpt${LATEST_CHECKPOINT}/summary.csv
  ${EVAL_DIR}/v9_ckpt${LATEST_CHECKPOINT}/summary.csv

EOF