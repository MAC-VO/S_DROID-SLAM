#!/bin/bash
set -euo pipefail

# Auto-detect GPU architecture and compile only for current GPU.
CUDA_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null) || {
    echo "Error: Failed to detect GPU architecture. Ensure CUDA is available."
    exit 1
}
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
echo "Compiling for GPU architecture: $CUDA_ARCH"

# Install lietorch (skip if already installed, e.g. from DPVO)
if ! python -c "import lietorch" 2>/dev/null; then
    echo "Installing lietorch..."
    pip install -v thirdparty/lietorch --no-build-isolation
else
    echo "lietorch already installed, skipping..."
fi

# Install droid-backends (CUDA extensions)
pip install -v -e . --no-build-isolation
