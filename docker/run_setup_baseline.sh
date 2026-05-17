#!/bin/bash
set -euo pipefail
# Submit a Run:AI job to prepare and optionally push the baseline checkpoint.
# Keeps all work under /scratch/workspace/henrik/standard-project-m2-mystery-machine.
#
# Usage (local):
#   ./docker/run_setup_baseline.sh --local [--repo <REPO>] [--dest <dest_rel>] [--push-repo <hf_repo>] [--commit-message "..."] --yes
#
# Usage (submit job):
#   ./docker/run_setup_baseline.sh                       # submit job that runs setup_baseline.py
#   ./docker/run_setup_baseline.sh --push-repo user/model # also push to HF model repo

# ============== EDIT THESE LINES ==============
GASPAR="hgruber"              # <-- YOUR GASPAR EPFL username.
GROUP="g02"                   # <-- YOUR TEAM, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit run_setup_baseline.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit run_setup_baseline.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

WORKSPACE_BASE="/scratch/workspace/henrik/standard-project-m2-mystery-machine"
SUFFIX="setup-baseline"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="ayushkumartarun/course-cs-552-standard:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Allow quick local run without submitting a job
if [[ "${1:-}" == "--local" ]]; then
  shift || true
  exec python scripts/setup_baseline.py "$@"
fi

# Build sanitized args for the remote run so --dest ends up under WORKSPACE_BASE
DEST_REL="models/qwen3_baseline"
NEW_ARGS=()
SKIP_NEXT=0
while (( "$#" )); do
  arg="$1"
  if [[ $SKIP_NEXT -eq 1 ]]; then
    SKIP_NEXT=0
    shift
    continue
  fi
  case "$arg" in
    --dest=*)
      DEST_REL="${arg#--dest=}"
      shift
      ;;
    --dest)
      DEST_REL="$2"
      SKIP_NEXT=1
      shift
      ;;
    *)
      NEW_ARGS+=("$arg")
      shift
      ;;
  esac
done

# Normalize dest to be relative and safe
DEST_REL="${DEST_REL#/}"         # strip leading /
if [[ "$DEST_REL" == *".."* ]]; then
  echo "Refusing dest containing '..'" >&2
  exit 1
fi
DEST_ABS="${WORKSPACE_BASE}/${DEST_REL}"

# Append normalized --dest to args
NEW_ARGS+=("--dest" "$DEST_ABS")

echo ">>> Submitting ${JOB_NAME} to prepare baseline (dest: ${DEST_ABS})"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu 0 \
  --large-shm \
  --node-pools "a100-40g" \
  --working-dir "${WORKSPACE_BASE}" \
  --environment HF_HOME="${WORKSPACE_BASE}/hf_cache" \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p '${WORKSPACE_BASE}' '${WORKSPACE_BASE}/hf_cache' '${WORKSPACE_BASE}/wandb' && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd '${WORKSPACE_BASE}' && \
    # clone the git repo into the workspace (idempotent) && \
    git clone https://github.com/$(git config --get remote.origin.url | sed -e 's#.*github.com/##' -e 's#\\.git$##') repo || true && \
    cd repo || true && \
    # Run setup_baseline.py with sanitized args (ensures --dest under workspace)
    python scripts/setup_baseline.py ${NEW_ARGS[@]/#/ }"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the pod is Running, use:
  runai bash ${JOB_NAME} -p ${PROJECT}
to get a shell inside the pod.
EOF
