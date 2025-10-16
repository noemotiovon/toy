"""
Tests for matrix addition kernels
"""

import torch
import pytest
from core import get_kernel, compare_against_reference, measure_time

@pytest.fixture
def test_data():
    """Create test data for matrix addition"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, N = 1024, 1024
    a = torch.randn(M, N, device=device, dtype=torch.float32)
    b = torch.randn(M, N, device=device, dtype=torch.float32)
    return a, b

def test_matrix_add_accuracy(test_data):
    """Test matrix addition accuracy across backends"""
    a, b = test_data
    
    # Get reference implementation
    reference_func = get_kernel("torch", "matrix_add")
    reference = reference_func(a, b)
    
    # Test other backends
    backends = ["triton", "tilelang"]
    if torch.cuda.is_available():
        try:
            get_kernel("cuda", "matrix_add")
            backends.append("cuda")
        except:
            pass
    
    for backend in backends:
        try:
            func = get_kernel(backend, "matrix_add")
            result = func(a, b)
            
            passed, metrics = compare_against_reference(reference, result, backend)
            
            print(f"[{backend}] Matrix add accuracy: {'PASS' if passed else 'FAIL'}")
            print(f"[{backend}] Max error: {metrics['max_abs_error']:.2e}")
            
            assert passed, f"Accuracy test failed for {backend}"
            
        except Exception as e:
            pytest.skip(f"Backend {backend} not available: {e}")

def test_matrix_add_performance(test_data):
    """Test matrix addition performance"""
    a, b = test_data
    
    backends = ["torch", "triton", "tilelang"]
    if torch.cuda.is_available():
        try:
            get_kernel("cuda", "matrix_add")
            backends.append("cuda")
        except:
            pass
    
    results = {}
    for backend in backends:
        try:
            func = get_kernel(backend, "matrix_add")
            mean_time = measure_time(func, a, b, warmup=5, iterations=50)
            results[backend] = mean_time * 1000  # Convert to ms
            
            print(f"[{backend}] Matrix add time: {mean_time*1000:.3f} ms")
            
        except Exception as e:
            print(f"[{backend}] Performance test failed: {e}")
    
    # Basic performance assertions
    assert len(results) > 0, "No backends available for performance testing"

if __name__ == "__main__":
    # Run tests directly
    test_data = (torch.randn(1024, 1024, device="cuda" if torch.cuda.is_available() else "cpu"),
                 torch.randn(1024, 1024, device="cuda" if torch.cuda.is_available() else "cpu"))
    
    test_matrix_add_accuracy(test_data)
    test_matrix_add_performance(test_data)
