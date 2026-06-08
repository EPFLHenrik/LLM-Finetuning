#!/bin/bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/workspace/emma/standard-project-m3-mystery-machine}"
cd "${REPO_DIR}"
source code/multilinguality_model/setup_env.sh

echo "Checking mergekit"

if ! command -v mergekit-yaml >/dev/null 2>&1; then
  python -m pip install mergekit
fi

python - <<'PY'
from pathlib import Path
import mergekit.architecture as arch

path = Path(arch.__file__)
text = path.read_text(encoding="utf-8")

# Some mergekit/pydantic versions fail to resolve torch.Tensor forward references
# inside ConfiguredModuleArchitecture/ConfiguredModelArchitecture. This patch is
# local to the current pod and rebuilds the pydantic models with torch available.

if "import torch" not in text:
    text = "import torch\n" + text

patch = """
try:
    ConfiguredModuleArchitecture.model_rebuild(
        force=True,
        _types_namespace={"torch": torch},
    )
    ConfiguredModelArchitecture.model_rebuild(
        force=True,
        _types_namespace={"torch": torch},
    )
except Exception as exc:
    print(f"[mergekit patch warning] {exc}")
"""

if "ConfiguredModuleArchitecture.model_rebuild(" not in text:
    text = text.rstrip() + "\n\n" + patch + "\n"

path.write_text(text, encoding="utf-8")
print(f"patched {path}")
PY

command -v mergekit-yaml >/dev/null 2>&1
mergekit-yaml --help >/dev/null

echo "mergekit ready"
