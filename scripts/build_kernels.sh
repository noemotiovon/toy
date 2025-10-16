#!/bin/bash

# Build script for CUDA kernels

set -e

echo "Building high-performance kernels..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Check if PyTorch is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -name "*.so" -delete

# Build CUDA extensions
echo "Building CUDA extensions..."
python setup.py build_ext --inplace

# Verify build
echo "Verifying build..."
python -c "
try:
    import kernels_cuda
    print('✓ CUDA kernels built successfully')
    print(f'Available functions: {dir(kernels_cuda)}')
except ImportError as e:
    print(f'✗ CUDA kernels build failed: {e}')
    exit(1)
"

echo "Build completed successfully!"
