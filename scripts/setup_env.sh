#!/bin/bash
# GriNNder environment setup script
# Usage: bash scripts/setup_env.sh [env_name] [cuda_version]
# Example: bash scripts/setup_env.sh grinnder cu124

set -e

ENV_NAME="${1:-grinnder}"
CUDA="${2:-cu124}"
PYTHON_VERSION="3.11"

echo "=== GriNNder Environment Setup ==="
echo "  Environment: ${ENV_NAME}"
echo "  CUDA: ${CUDA}"
echo "  Python: ${PYTHON_VERSION}"
echo ""

# 1. Create conda environment
echo "[1/6] Creating conda environment..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "${ENV_NAME}"

# 2. Install PyTorch
echo "[2/6] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA}"

# 3. Install PyG, OGB, test tooling, and sparse extensions
echo "[3/6] Installing PyTorch Geometric, OGB, pytest, and sparse extensions..."
pip install torch-geometric numpy ogb pytest
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-sparse torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA}.html"

# 4. Install kvikio (GPUDirect Storage)
echo "[4/6] Installing kvikio..."
pip install kvikio-cu12

# 5. Install grdpart (graph partitioning)
echo "[5/6] Installing grdpart..."
if [ -d "third_party/grdpart" ]; then
    pip install third_party/grdpart/ --no-build-isolation
elif command -v git &>/dev/null; then
    TMPDIR=$(mktemp -d)
    git clone https://github.com/AIS-SNU/grdpart.git "${TMPDIR}/grdpart"
    pip install "${TMPDIR}/grdpart/" --no-build-isolation
    rm -rf "${TMPDIR}"
else
    pip install grdpart
fi

# 6. Build and install GriNNder
echo "[6/6] Building GriNNder (C++ extensions + io_uring)..."

# Build bundled liburing if not built yet
if [ ! -f "third_party/liburing/src/liburing.a" ]; then
    echo "  Building bundled liburing 2.8..."
    cd third_party/liburing
    ./configure --prefix=/usr/local
    make -j$(nproc)
    cd ../..
fi

# Find a compatible system CUDA (12.x preferred, must have nvcc)
TORCH_CUDA=$(python -c "import torch; v=torch.version.cuda; print(v[:2] if v else '12')")
CUDA_HOME=""
for d in /usr/local/cuda-${TORCH_CUDA}.* /usr/local/cuda-${TORCH_CUDA} /usr/local/cuda; do
    if [ -x "$d/bin/nvcc" ]; then
        CUDA_HOME="$d"
        break
    fi
done
if [ -z "$CUDA_HOME" ]; then
    echo "  WARNING: No system CUDA ${TORCH_CUDA}.x found, using default nvcc"
else
    echo "  Using CUDA_HOME=$CUDA_HOME"
    export CUDA_HOME="$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# Detect CUDA architectures for all visible GPUs. Override with
# TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' when building a portable wheel.
GPU_ARCHS=$(python -c "
import torch
if torch.cuda.is_available():
    archs = []
    for i in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(i)
        arch = f'{cc[0]}.{cc[1]}'
        if arch not in archs:
            archs.append(arch)
    print(';'.join(archs))
else:
    print('8.0')
" 2>/dev/null)
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${CUDA_ARCH_LIST:-${GPU_ARCHS}}}"
echo "  CUDA architectures: ${CUDA_ARCH_LIST}"

TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" pip install -e . --no-build-isolation

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""

# Verify installation
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
# Quick matmul sanity check
a = torch.ones(2, 2, device='cuda')
assert (torch.mm(a, a) == 2.0).all(), 'CUBLAS broken!'
print('CUBLAS: OK')
import torch_geometric
print(f'PyG {torch_geometric.__version__}')
import ogb
print(f'OGB {ogb.__version__}')
import grinnder
print(f'GriNNder {grinnder.__version__}')
from grinnder._C import IoUringEngine
engine = IoUringEngine(4)
print(f'io_uring: {engine.has_io_uring()}')
import kvikio
print('kvikio: available')
print()
print('All components OK')
"
