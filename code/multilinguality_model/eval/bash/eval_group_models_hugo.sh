#!/bin/bash
set -euo pipefail

HUGO_REPO="/scratch/workspace/hugo/standard-project-m3-mystery-machine"
GROUP_MODEL_ROOT="${HUGO_REPO}/checkpoints/group_model"

WANDB_ENTITY="hugo-demule-epfl"
WANDB_PROJECT="MNLP-Group-Model-Benchmarks"

MULTILINGUAL_MODELS=(
  "all_full_ta_multilingual_v1"
  "all_full_ta_multilingual_v13"
  "all_full_ta_multilingual_v16"
  "all_full_ta_multilingual_v19"
)

cd "${HUGO_REPO}"
export PYTHONPATH="${HUGO_REPO}:${PYTHONPATH:-}"

echo "============================================================"
echo "Multilingual model benchmark evaluation"
echo "Hugo repo:        ${HUGO_REPO}"
echo "Group model root: ${GROUP_MODEL_ROOT}"
echo "W&B entity:       ${WANDB_ENTITY}"
echo "W&B project:      ${WANDB_PROJECT}"
echo "Time:             $(date)"
echo "============================================================"

echo ""
echo "Checking GPU visibility..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found. Run this from a GPU node."
  exit 1
fi

nvidia-smi

python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] No CUDA device visible.")
PY

echo ""
echo "Checking for existing vLLM engine processes..."
if nvidia-smi | grep -q "VLLM::EngineCore"; then
  echo "[ERROR] Existing VLLM::EngineCore process is already using the GPU."
  exit 1
fi

if [[ ! -d "${HUGO_REPO}/data/group_model/benchmark" ]]; then
  echo "[ERROR] Benchmark folder not found: ${HUGO_REPO}/data/group_model/benchmark"
  exit 1
fi

FAILED=()

for NAME in "${MULTILINGUAL_MODELS[@]}"; do
  CKPT="${GROUP_MODEL_ROOT}/${NAME}"

  if [[ ! -d "${CKPT}" ]]; then
    echo "[SKIP] ${NAME} — directory not found"
    continue
  fi

  if [[ ! -f "${CKPT}/config.json" ]]; then
    echo "[SKIP] ${NAME} — missing config.json"
    continue
  fi

  echo ""
  echo "============================================================"
  echo "Evaluating: ${NAME}"
  echo "Checkpoint: ${CKPT}"
  echo "Time:       $(date)"
  echo "============================================================"

  set +e
  python -m code.group_model.benchmark \
    --checkpoint "${CKPT}" \
    --run_all \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --name "${NAME}" \
    --limit 100
  status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[ERROR] benchmark.py failed for ${NAME} with exit code ${status}"
    FAILED+=("${NAME}")
  else
    echo "Finished ${NAME} at $(date)"
  fi

  echo ""
  echo "Checking for stale vLLM processes after ${NAME}..."
  if nvidia-smi | grep -q "VLLM::EngineCore"; then
    echo "[WARN] VLLM::EngineCore still present, waiting 10s..."
    sleep 10
  fi

  if nvidia-smi | grep -q "VLLM::EngineCore"; then
    echo "[ERROR] Stale VLLM::EngineCore still holds GPU memory. Stopping."
    FAILED+=("${NAME}: stale_vllm")
    break
  fi
done

echo ""
echo "============================================================"
echo "Multilingual evaluation finished"
echo "Time: $(date)"
echo "============================================================"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed runs:"
  for item in "${FAILED[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi

echo "All multilingual model evaluations completed successfully."