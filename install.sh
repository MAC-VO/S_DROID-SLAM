#!/bin/bash
set -euo pipefail

if [[ "${1:-}" == "--force" ]]; then
    FORCE_REBUILD=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect GPU architecture and compile only for current GPU.
CUDA_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null) || {
    echo "Error: Failed to detect GPU architecture. Ensure CUDA is available."
    exit 1
}
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
echo "Compiling for GPU architecture: $CUDA_ARCH"

# `-P` so Python doesn't auto-prepend CWD onto sys.path; otherwise the source
# tree masks a missing pip install and the check always reports installed.
if python -P -c "import droid_backends" 2>/dev/null && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[DROID_SLAM] Already installed, skipping. Use --force to reinstall."
    exit 0
fi

# Install lietorch (skip if already installed, e.g. from DPVO)
if ! python -P -c "import lietorch" 2>/dev/null; then
    echo "Installing lietorch..."
    pip install -v "$SCRIPT_DIR/thirdparty/lietorch" --no-build-isolation
else
    echo "lietorch already installed, skipping..."
fi

# Install droid-backends (CUDA extensions)
cd "$SCRIPT_DIR"
pip install -v -e . --no-build-isolation
