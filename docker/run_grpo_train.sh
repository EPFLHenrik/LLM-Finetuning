#!/usr/bin/env bash
set -euo pipefail

# Submit a Run:AI job to run GRPO training on the cluster.
# Mirrors the style of other scripts in this folder and supports a local
# quick-run mode with `--local`.

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_grpo_train.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_grpo_train.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

SUFFIX="grpo-train"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# ---- Mandatory (edit before submitting) ----
MODEL_FOLDER="models/qwen3_thinking"                              # REQUIRED: base model to train from
DATASET_FILE="data/preprocessing/wrong_distillations.jsonl"      # REQUIRED: path to .jsonl or .json
OUTPUT_DIR="models/grpo_output"                                   # REQUIRED: where merged checkpoint will be written

# ---- Training ----
SEED=42
NUM_TRAIN_EPOCHS=1.0
LEARNING_RATE=3e-6
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LOGGING_STEPS=80
SAVE_STRATEGY="epoch"
SAVE_STEPS=500
MAX_TRAIN_SAMPLES=0          # 0 = use all samples

# ---- GRPO / generation ----
NUM_GENERATIONS=8
MAX_COMPLETION_LENGTH=4096
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=0
REPETITION_PENALTY=1.0

# ---- vLLM ----
VLLM_GPU_MEMORY_UTILIZATION=0.35
VLLM_TENSOR_PARALLEL_SIZE=1

# ---- LoRA ----
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# ---- Optional: load a PEFT adapter on top of a base model ----
PEFT_ADAPTER=""              # e.g. models/my_lora_adapter  (leave empty to skip)
BASE_MODEL=""                # required when PEFT_ADAPTER is set

# ---- Flags ----
GRADIENT_CHECKPOINTING=true
SKIP_WANDB=false
OVERWRITE_OUTPUT=true
DRY_RUN=false

# ---- W&B ----
WANDB_PROJECT="cs552-mystery-machine"
WANDB_ENTITY=""
WANDB_KEY="${WANDB_API_KEY:-}"

# ---- Build flag arguments ----
GRADIENT_CHECKPOINTING_ARG="--gradient-checkpointing"
SKIP_WANDB_ARG=""
OVERWRITE_OUTPUT_ARG=""
DRY_RUN_ARG=""
NO_LORA_ARG=""
PEFT_ADAPTER_ARG=""
BASE_MODEL_ARG=""

if [ "$GRADIENT_CHECKPOINTING" = "false" ]; then GRADIENT_CHECKPOINTING_ARG="--no-gradient-checkpointing"; fi
if [ "$SKIP_WANDB" = "true" ];             then SKIP_WANDB_ARG="--skip-wandb"; fi
if [ "$OVERWRITE_OUTPUT" = "true" ];       then OVERWRITE_OUTPUT_ARG="--overwrite-output"; fi
if [ "$DRY_RUN" = "true" ];               then DRY_RUN_ARG="--dry-run"; fi
if [ "$USE_LORA" = "false" ];             then NO_LORA_ARG="--no-lora"; fi
if [ -n "$PEFT_ADAPTER" ];                then PEFT_ADAPTER_ARG="--peft-adapter \"${PEFT_ADAPTER}\""; fi
if [ -n "$BASE_MODEL" ];                  then BASE_MODEL_ARG="--base-model \"${BASE_MODEL}\""; fi

# ---- Validate mandatory variables ----
if [ -z "$MODEL_FOLDER" ] || [ -z "$DATASET_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "ERROR: set MODEL_FOLDER, DATASET_FILE, and OUTPUT_DIR near the top of this script before submitting." >&2
  exit 1
fi

echo ">>> Submitting ${JOB_NAME} (GRPO training job)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu 1 \
  --large-shm \
  --node-pools "a100-40g" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment WANDB_DIR=/scratch/wandb \
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment PYTORCH_ALLOC_CONF=expandable_segments:True \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"${REPO_DIR:-/scratch/workspace/henrik/standard-project-m2-mystery-machine}\" && \
    python training/grpo_train.py \
      --model-folder \"${MODEL_FOLDER}\" \
      --dataset-file \"${DATASET_FILE}\" \
      --output-dir \"${OUTPUT_DIR}\" \
      --seed ${SEED} \
      --num-train-epochs ${NUM_TRAIN_EPOCHS} \
      --learning-rate ${LEARNING_RATE} \
      --per-device-train-batch-size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
      --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
      --logging-steps ${LOGGING_STEPS} \
      --save-strategy \"${SAVE_STRATEGY}\" \
      --save-steps ${SAVE_STEPS} \
      --max-train-samples ${MAX_TRAIN_SAMPLES} \
      --num-generations ${NUM_GENERATIONS} \
      --max-completion-length ${MAX_COMPLETION_LENGTH} \
      --temperature ${TEMPERATURE} \
      --top-p ${TOP_P} \
      --top-k ${TOP_K} \
      --repetition-penalty ${REPETITION_PENALTY} \
      --vllm-gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
      --vllm-tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE} \
      --lora-r ${LORA_R} \
      --lora-alpha ${LORA_ALPHA} \
      --lora-dropout ${LORA_DROPOUT} \
      --lora-target-modules \"${LORA_TARGET_MODULES}\" \
      --wandb-project \"${WANDB_PROJECT}\" \
      --wandb-entity \"${WANDB_ENTITY}\" \
      --wandb-key \"${WANDB_KEY}\" \
      ${GRADIENT_CHECKPOINTING_ARG} \
      ${NO_LORA_ARG} \
      ${PEFT_ADAPTER_ARG} \
      ${BASE_MODEL_ARG} \
      ${SKIP_WANDB_ARG} \
      ${OVERWRITE_OUTPUT_ARG} \
      ${DRY_RUN_ARG}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF
