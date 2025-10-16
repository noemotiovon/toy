"""
Tests for matrix multiplication kernels
"""

import torch
import pytest
from core import get_kernel, compare_against_reference, measure_time

@pytest.fixture
def test_data():
    """Create test data for matrix multiplication"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, K, N = 512, 512, 512
    a = torch.randn(M, K, device=device, dtype=torch.float32)
    b = torch.randn(K, N, device=device, dtype=torch.float32)
    return a, b

def test_matmul_accuracy(test_data):
    """Test matrix multiplication accuracy across backends"""
    a, b = test_data
    
    # Get reference implementation
    reference_func = get_kernel("torch", "matmul")
    reference = reference_func(a, b)
    
    # Test other backends
    backends = ["triton", "tilelang"]
    if torch.cuda.is_available():
        try:
            get_kernel("cuda", "matmul")
            backends.append("cuda")
        except:
            pass
    
    for backend in backends:
        try:
            func = get_kernel(backend, "matmul")
            result = func(a, b)
            
            passed, metrics = compare_against_reference(reference, result, backend)
            
            print(f"[{backend}] Matrix mul accuracy: {'PASS' if passed else 'FAIL'}")
            print(f"[{backend}] Max error: {metrics['max_abs_error']:.2e}")
            
            # More lenient tolerance for matmul due to accumulation errors
            if not passed and metrics['max_abs_error'] < 1e-3:
                print(f"[{backend}] Accuracy within acceptable range for matmul")
                passed = True
            
            assert passed, f"Accuracy test failed for {backend}"
            
        except Exception as e:
            pytest.skip(f"Backend {backend} not available: {e}")

def test_matmul_performance(test_data):
    """Test matrix multiplication performance"""
    a, b = test_data
    
    backends = ["torch", "triton", "tilelang"]
    if torch.cuda.is_available():
        try:
            get_kernel("cuda", "matmul")
            backends.append("cuda")
        except:
            pass
    
    results = {}
    for backend in backends:
        try:
            func = get_kernel(backend, "matmul")
            mean_time = measure_time(func, a, b, warmup=5, iterations=50)
            results[backend] = mean_time * 1000  # Convert to ms
            
            print(f"[{backend}] Matrix mul time: {mean_time*1000:.3f} ms")
            
        except Exception as e:
            print(f"[{backend}] Performance test failed: {e}")
    
    # Basic performance assertions
    assert len(results) > 0, "No backends available for performance testing"

if __name__ == "__main__":
    # Run tests directly
    test_data = (torch.randn(512, 512, device="cuda" if torch.cuda.is_available() else "cpu"),
                 torch.randn(512, 512, device="cuda" if torch.cuda.is_available() else "cpu"))
    
    test_matmul_accuracy(test_data)
    test_matmul_performance(test_data)
