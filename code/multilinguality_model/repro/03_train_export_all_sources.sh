#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "${REPO_DIR}"

bash code/multilinguality_model/repro/02_train_export_source_model.sh v1
bash code/multilinguality_model/repro/02_train_export_source_model.sh v20-russian
bash code/multilinguality_model/repro/02_train_export_source_model.sh v20-hindi
