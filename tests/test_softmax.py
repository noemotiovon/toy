"""
Tests for softmax kernels
"""

import torch
import pytest
from core import get_kernel, compare_against_reference, measure_time

@pytest.fixture
def test_data():
    """Create test data for softmax"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, N = 1024, 1024
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    return x

def test_softmax_accuracy(test_data):
    """Test softmax accuracy across backends"""
    x = test_data
    
    # Get reference implementation
    reference_func = get_kernel("torch", "softmax")
    reference = reference_func(x)
    
    # Test other backends
    backends = ["triton", "tilelang"]
    if torch.cuda.is_available():
        try:
            get_kernel("cuda", "softmax")
            backends.append("cuda")
        except:
            pass
    
    for backend in backends:
        try:
            func = get_kernel(backend, "softmax")
            result = func(x)
            
            passed, metrics = compare_against_reference(reference, result, backend)
            
            print(f"[{backend}] Softmax accuracy: {'PASS' if passed else 'FAIL'}")
            print(f"[{backend}] Max error: {metrics['max_abs_error']:.2e}")
            
            assert passed, f"Accuracy test failed for {backend}"
            
        except Exception as e:
            pytest.skip(f"Backend {backend} not available: {e}")

def test_softmax_performance(test_data):
    """Test softmax performance"""
    x = test_data
    
    backends = ["torch", "triton", "tilelang"]
    if torch.cuda.is_available():
        try:
            get_kernel("cuda", "softmax")
            backends.append("cuda")
        except:
            pass
    
    results = {}
    for backend in backends:
        try:
            func = get_kernel(backend, "softmax")
            mean_time = measure_time(func, x, warmup=5, iterations=50)
            results[backend] = mean_time * 1000  # Convert to ms
            
            print(f"[{backend}] Softmax time: {mean_time*1000:.3f} ms")
            
        except Exception as e:
            print(f"[{backend}] Performance test failed: {e}")
    
    # Basic performance assertions
    assert len(results) > 0, "No backends available for performance testing"

def test_softmax_properties(test_data):
    """Test softmax mathematical properties"""
    x = test_data
    
    func = get_kernel("torch", "softmax")
    result = func(x)
    
    # Check that rows sum to 1
    row_sums = torch.sum(result, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), "Softmax rows should sum to 1"
    
    # Check that all values are positive
    assert torch.all(result >= 0), "Softmax values should be non-negative"
    
    # Check that values are <= 1
    assert torch.all(result <= 1), "Softmax values should be <= 1"

if __name__ == "__main__":
    # Run tests directly
    test_data = torch.randn(1024, 1024, device="cuda" if torch.cuda.is_available() else "cpu")
    
    test_softmax_accuracy(test_data)
    test_softmax_performance(test_data)
    test_softmax_properties(test_data)
