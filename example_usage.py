#!/usr/bin/env python3
"""
Example usage of the CUDA-Triton-TileLang comparison project.
This script demonstrates how to use the different implementations.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_matrix_add():
    """Example of matrix addition using different implementations"""
    print("Matrix Addition Example")
    print("=" * 30)
    
    # Create test data
    M, N = 1024, 1024
    a = torch.randn(M, N, device='cuda' if torch.cuda.is_available() else 'cpu')
    b = torch.randn(M, N, device=a.device)
    
    print(f"Matrix size: {M}x{N}")
    print(f"Device: {a.device}")
    
    # PyTorch reference
    try:
        from pytorch_ops import matrix_add as pytorch_matrix_add
        pytorch_result = pytorch_matrix_add(a, b)
        print("✓ PyTorch implementation completed")
    except Exception as e:
        print(f"✗ PyTorch implementation failed: {e}")
        return
    
    # Triton implementation
    try:
        from triton_ops import matrix_add as triton_matrix_add
        triton_result = triton_matrix_add(a, b)
        triton_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
        print(f"✓ Triton implementation completed (error: {triton_error:.2e})")
    except Exception as e:
        print(f"✗ Triton implementation failed: {e}")
    
    # TileLang implementation
    try:
        from tilelang_ops import matrix_add as tilelang_matrix_add
        tilelang_result = tilelang_matrix_add(a, b)
        tilelang_error = torch.max(torch.abs(pytorch_result - tilelang_result)).item()
        print(f"✓ TileLang implementation completed (error: {tilelang_error:.2e})")
    except Exception as e:
        print(f"✗ TileLang implementation failed: {e}")

def example_matrix_mul():
    """Example of matrix multiplication using different implementations"""
    print("\nMatrix Multiplication Example")
    print("=" * 30)
    
    # Create test data
    M, K, N = 512, 512, 512
    a = torch.randn(M, K, device='cuda' if torch.cuda.is_available() else 'cpu')
    b = torch.randn(K, N, device=a.device)
    
    print(f"Matrix sizes: {M}x{K} @ {K}x{N}")
    print(f"Device: {a.device}")
    
    # PyTorch reference
    try:
        from pytorch_ops import matrix_mul as pytorch_matrix_mul
        pytorch_result = pytorch_matrix_mul(a, b)
        print("✓ PyTorch implementation completed")
    except Exception as e:
        print(f"✗ PyTorch implementation failed: {e}")
        return
    
    # Triton implementation
    try:
        from triton_ops import matrix_mul as triton_matrix_mul
        triton_result = triton_matrix_mul(a, b)
        triton_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
        print(f"✓ Triton implementation completed (error: {triton_error:.2e})")
    except Exception as e:
        print(f"✗ Triton implementation failed: {e}")
    
    # TileLang implementation
    try:
        from tilelang_ops import matrix_mul as tilelang_matrix_mul
        tilelang_result = tilelang_matrix_mul(a, b)
        tilelang_error = torch.max(torch.abs(pytorch_result - tilelang_result)).item()
        print(f"✓ TileLang implementation completed (error: {tilelang_error:.2e})")
    except Exception as e:
        print(f"✗ TileLang implementation failed: {e}")

def example_softmax():
    """Example of softmax using different implementations"""
    print("\nSoftmax Example")
    print("=" * 30)
    
    # Create test data
    M, N = 1024, 1024
    x = torch.randn(M, N, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Input size: {M}x{N}")
    print(f"Device: {x.device}")
    
    # PyTorch reference
    try:
        from pytorch_ops import softmax as pytorch_softmax
        pytorch_result = pytorch_softmax(x)
        print("✓ PyTorch implementation completed")
    except Exception as e:
        print(f"✗ PyTorch implementation failed: {e}")
        return
    
    # Triton implementation
    try:
        from triton_ops import softmax as triton_softmax
        triton_result = triton_softmax(x)
        triton_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
        print(f"✓ Triton implementation completed (error: {triton_error:.2e})")
    except Exception as e:
        print(f"✗ Triton implementation failed: {e}")
    
    # TileLang implementation
    try:
        from tilelang_ops import softmax as tilelang_softmax
        tilelang_result = tilelang_softmax(x)
        tilelang_error = torch.max(torch.abs(pytorch_result - tilelang_result)).item()
        print(f"✓ TileLang implementation completed (error: {tilelang_error:.2e})")
    except Exception as e:
        print(f"✗ TileLang implementation failed: {e}")

def main():
    """Run all examples"""
    print("CUDA-Triton-TileLang Comparison Examples")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    print()
    
    # Run examples
    example_matrix_add()
    example_matrix_mul()
    example_softmax()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run full tests, use:")
    print("python tests/run_all_tests.py")

if __name__ == "__main__":
    main()
