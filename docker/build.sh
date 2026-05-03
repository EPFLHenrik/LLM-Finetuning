#!/bin/bash
# Build & push a CS-552 custom image to your own public Docker Hub repo.
# Run from this directory.
#
# Most teams should use the default course image directly and do not need
# this script. It is only for teams that need a custom image with extra
# system packages or libraries.
#
# Prereqs:
#   1. A Docker Hub account with a PUBLIC repo you own
#      (see https://docs.docker.com/docker-hub/quickstart/ and
#       https://docs.docker.com/docker-hub/repos/create/).
#   2. Run `docker login` once with your Docker Hub credentials.
#   3. Edit DOCKERHUB_USER, IMAGE, and DOCKERFILE below.
#
# Usage:
#   ./build.sh           # build & push :v1
#   ./build.sh v2        # build & push :v2
#
# After it finishes, copy the printed image name into submit.sh as IMAGE=.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============== EDIT THESE LINES ==============
DOCKERHUB_USER="dockerhub-user"     # <-- YOUR Docker Hub username.
IMAGE="cs552-custom"                # <-- name of your public repo on Docker Hub.
DOCKERFILE="Dockerfile"             # <-- path to your Dockerfile (relative to this script)
# ==============================================

# Refuse to run with the placeholder.
if [[ "${DOCKERHUB_USER}" == "dockerhub-user" || -z "${DOCKERHUB_USER}" ]]; then
    echo "ERROR: edit build.sh and set DOCKERHUB_USER to your Docker Hub username." >&2
    exit 1
fi

REGISTRY="docker.io"
TAG="${1:-v1}"

# Only used by Dockerfile.open (vllm-openai base). Ignored otherwise.
# Pin to a specific upstream tag, never `:latest`.
VLLM_TAG="${VLLM_TAG:-v0.11.0}"

FULL="${REGISTRY}/${DOCKERHUB_USER}/${IMAGE}:${TAG}"

echo ">>> Building ${FULL} from ${DOCKERFILE}"
docker build \
  --pull \
  --platform linux/amd64 \
  -f "${SCRIPT_DIR}/${DOCKERFILE}" \
  --build-arg "VLLM_TAG=${VLLM_TAG}" \
  -t "${FULL}" \
  "${SCRIPT_DIR}"

echo ">>> Pushing ${FULL}"
docker push "${FULL}"

echo ">>> Done. In submit.sh, set:"
echo "    IMAGE=\"${FULL}\""
