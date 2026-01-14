#!/bin/bash
set -euo pipefail

# Install lietorch (skip if already installed, e.g. from DPVO)
if ! python -c "import lietorch" 2>/dev/null; then
    echo "Installing lietorch..."
    pip install -v thirdparty/lietorch --no-build-isolation
else
    echo "lietorch already installed, skipping..."
fi

# Install droid-backends (CUDA extensions)
pip install -v -e . --no-build-isolation
