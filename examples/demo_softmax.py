#!/usr/bin/env python3
"""
Softmax Kernel Demo

This script demonstrates the softmax kernel across different backends
and shows accuracy comparison against PyTorch reference.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from core import get_kernel, get_available_backends, compare_against_reference, measure_time

def main():
    print("Softmax Kernel Demo")
    print("=" * 40)
    
    # Get available backends
    backends = get_available_backends()
    print(f"Available backends: {backends}")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test parameters
    M, N = 512, 1024
    print(f"Matrix size: {M}x{N}")
    
    # Create test data
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    print(f"Data type: {x.dtype}")
    print()
    
    # Get reference implementation
    reference_kernel = get_kernel('torch', 'softmax')
    print("Running reference implementation (PyTorch)...")
    reference_result = reference_kernel(x)
    print()
    
    # Test other backends
    print("Testing other backends:")
    print("-" * 30)
    print()
    
    for backend in backends:
        if backend == 'torch':
            continue
            
        print(f"[{backend.upper()}]")
        try:
            kernel = get_kernel(backend, 'softmax')
            result = kernel(x)
            
            # Compare accuracy
            passed, metrics = compare_against_reference(reference_result, result, backend_name=backend)
            
            # Measure time
            times = measure_time(lambda: kernel(x), warmup=10, iterations=100)
            avg_time = np.mean(times) * 1000  # Convert to ms
            
            print(f"  Accuracy: {'✓ PASS' if passed else '✗ FAIL'}")
            print(f"  Max error: {metrics['max_abs_error']:.2e}")
            print(f"  Mean error: {metrics['mean_abs_error']:.2e}")
            print(f"  Time: {avg_time:.3f} ms")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    print("Demo completed!")

if __name__ == "__main__":
    main()
