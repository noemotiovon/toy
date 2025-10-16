#!/usr/bin/env python3
"""
Demo script for matrix multiplication kernels
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core import get_kernel, get_available_backends, compare_against_reference, measure_time

def main():
    print("Matrix Multiplication Kernel Demo")
    print("=" * 40)
    
    # Check available backends
    available_backends = get_available_backends()
    print(f"Available backends: {available_backends}")
    
    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    M, K, N = 512, 512, 512
    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)
    
    print(f"Matrix sizes: {M}x{K} @ {K}x{N}")
    print(f"Data type: {a.dtype}")
    
    # Get reference implementation
    print("\nRunning reference implementation (PyTorch)...")
    reference_func = get_kernel("torch", "matmul")
    reference = reference_func(a, b)
    
    # Test other backends
    print("\nTesting other backends:")
    print("-" * 30)
    
    for backend in available_backends:
        if backend == "torch":
            continue
            
        try:
            print(f"\n[{backend.upper()}]")
            
            # Get kernel
            func = get_kernel(backend, "matmul")
            
            # Test accuracy
            result = func(a, b)
            passed, metrics = compare_against_reference(reference, result, backend)
            
            print(f"  Accuracy: {'✓ PASS' if passed else '✗ FAIL'}")
            print(f"  Max error: {metrics['max_abs_error']:.2e}")
            print(f"  Mean error: {metrics['mean_abs_error']:.2e}")
            
            # For matmul, be more lenient with accuracy
            if not passed and metrics['max_abs_error'] < 1e-3:
                print(f"  → Accuracy within acceptable range for matmul")
                passed = True
            
            # Test performance
            mean_time = measure_time(func, a, b, warmup=5, iterations=50)
            print(f"  Time: {mean_time*1000:.3f} ms")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
