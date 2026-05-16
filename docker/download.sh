#!/bin/bash
# CS-552 — submit a non-interactive Run:AI job to download and preprocess
# a Hugging Face dataset into the repository-local data/ directory.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                    # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit download.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit download.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-download}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running download.sh}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH before running download.sh}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
QUESTION_KEYS="${QUESTION_KEYS:-question,question__stem,stem,query,text,prompt}"
CHOICE_KEYS="${CHOICE_KEYS:-choices,options,answer_choices,candidates}"
ANSWER_KEYS="${ANSWER_KEYS:-answerKey,answer_key,label,answer,answerIndex}"

REPO_DIR="/scratch/workspace/henrik/standard-project-m2-mystery-machine"
DATASET_SLUG="${DATASET_SLUG:-$(python - <<'PY'
import os
dataset_path = os.environ["DATASET_PATH"]
dataset_config = os.environ.get("DATASET_CONFIG") or ""
slug = dataset_path.replace("/", "__").replace(":", "__")
if dataset_config:
  slug += "__" + dataset_config.replace("/", "__").replace(":", "__").replace(" ", "_")
print(slug)
PY
)}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${REPO_DIR}/data/datasets/${DATASET_SLUG}}"
PREPROCESS_DATA_DIR="${PREPROCESS_DATA_DIR:-${REPO_DIR}/data/preprocessing/${DATASET_SLUG}}"

echo ">>> Submitting ${JOB_NAME} (1 GPU)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment WANDB_DIR=/scratch/wandb \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/data" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb \"${RAW_DATA_DIR}\" \"${PREPROCESS_DATA_DIR}\" && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"${REPO_DIR}\" && \
    python scripts/download_hf.py --hf-token \"${HF_TOKEN}\" --dataset-path \"${DATASET_PATH}\" ${DATASET_CONFIG:+--dataset-config \"${DATASET_CONFIG}\"} --output-dir \"${RAW_DATA_DIR}\" && \
    python scripts/preprocess_hf.py --dataset-path \"${DATASET_PATH}\" ${DATASET_CONFIG:+--dataset-config \"${DATASET_CONFIG}\"} --input-dir \"${RAW_DATA_DIR}\" --output-dir \"${PREPROCESS_DATA_DIR}\" --question-keys \"${QUESTION_KEYS}\" --choice-keys \"${CHOICE_KEYS}\" --answer-keys \"${ANSWER_KEYS}\""

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

Output locations inside the pod:
  ${RAW_DATA_DIR}
  ${PREPROCESS_DATA_DIR}

The job runs both Python files in order:
  1. scripts/download_hf.py
  2. scripts/preprocess_hf.py
EOF
