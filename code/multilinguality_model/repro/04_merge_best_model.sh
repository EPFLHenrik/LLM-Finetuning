#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "${REPO_DIR}"
source code/multilinguality_model/setup_env.sh

if ! command -v mergekit-yaml >/dev/null 2>&1; then
  echo "mergekit-yaml not found. Run 00_install_repro_requirements.sh first." >&2
  exit 1
fi

CONFIG="code/multilinguality_model/configs/merging/best_model_ties.yaml"
RAW_DIR="checkpoints/multilinguality_model/best_model_raw"
FINAL_DIR="checkpoints/multilinguality_model/best_model"
CHAT_TEMPLATE="code/multilinguality_model/assets/chat_template_multilingual-nonthinking.jinja"

for model_dir in \
  "checkpoints/multilinguality_model/v1" \
  "checkpoints/multilinguality_model/v20-russian" \
  "checkpoints/multilinguality_model/v20-hindi"; do
  if [[ ! -d "${model_dir}" ]]; then
    echo "Missing exported model: ${model_dir}" >&2
    exit 1
  fi
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing merge config: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
  echo "Missing chat template: ${CHAT_TEMPLATE}" >&2
  exit 1
fi

echo "Merging models into temporary directory"
rm -rf "${RAW_DIR}" "${FINAL_DIR}"
mkdir -p "$(dirname "${RAW_DIR}")"

mergekit-yaml "${CONFIG}" "${RAW_DIR}" \
  --copy-tokenizer \
  --trust-remote-code

echo "Finalizing merged model"

python -m multilinguality_model.train.export_full \
  --model_path "${RAW_DIR}" \
  --output_dir "${FINAL_DIR}" \
  --run_name best_model \
  --chat_template "${CHAT_TEMPLATE}" \
  --overwrite \
  --skip_model_load

rm -rf "${RAW_DIR}"

echo "Merged model saved to ${FINAL_DIR}"