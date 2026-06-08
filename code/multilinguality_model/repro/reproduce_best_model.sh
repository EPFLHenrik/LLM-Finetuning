#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "${REPO_DIR}"
mkdir -p logs

bash code/multilinguality_model/repro/00_install_repro_requirements.sh
bash code/multilinguality_model/repro/01_sync_repro_data.sh
bash code/multilinguality_model/repro/03_train_export_all_sources.sh
bash code/multilinguality_model/repro/04_merge_best_model.sh
bash code/multilinguality_model/repro/05_eval_best_model.sh
