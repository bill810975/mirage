#!/bin/bash
# MTP Test Runner
# Usage: ./run_all_tests.sh [--gpu]
#
# Without --gpu: runs local-only tests (no CUDA required)
# With --gpu:    also builds and runs GPU kernel tests

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "============================================"
echo " MTP Test Suite"
echo " Repo: $REPO_ROOT"
echo "============================================"

# Phase 1: Local tests (no GPU)
echo ""
echo "=== Phase 1: Verification + MLA reference tests ==="
python -m pytest "$SCRIPT_DIR/test_mtp_verification.py" -v --tb=short
echo "Phase 1: PASSED"

# Phase 2: GPU tests (optional)
if [[ "$1" == "--gpu" ]]; then
    echo ""
    echo "=== Phase 2: Building MLA CUDA kernel extension ==="
    cd "$SCRIPT_DIR"

    # Detect CUDA arch
    CUDA_ARCH=${CUDA_ARCH:-""}
    if [ -z "$CUDA_ARCH" ]; then
        # Try to auto-detect
        if python -c "import torch; print(torch.cuda.get_device_capability())" 2>/dev/null | grep -q "10, 0"; then
            CUDA_ARCH="10.0"
            echo "Detected SM100 (B200)"
        else
            echo "Auto-detect failed. Set CUDA_ARCH env var (e.g., CUDA_ARCH=8.0)"
            CUDA_ARCH="8.0"
        fi
    fi

    echo "Building with TORCH_CUDA_ARCH_LIST=$CUDA_ARCH"
    TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" pip install -e . 2>&1 | tail -5

    echo ""
    echo "=== Phase 2: Running MLA kernel tests ==="
    python "$SCRIPT_DIR/test_mla_kernel.py"
    echo "Phase 2: PASSED"
else
    echo ""
    echo "Skipping GPU tests. Run with --gpu to enable."
    echo "  Example: ./run_all_tests.sh --gpu"
fi

echo ""
echo "============================================"
echo " All tests PASSED!"
echo "============================================"
