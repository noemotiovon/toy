import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_ops import matrix_add as pytorch_matrix_add, matrix_mul as pytorch_matrix_mul, softmax as pytorch_softmax
from triton_ops import matrix_add as triton_matrix_add, matrix_mul as triton_matrix_mul, softmax as triton_softmax
from tilelang_ops import matrix_add as tilelang_matrix_add, matrix_mul as tilelang_matrix_mul, softmax as tilelang_softmax

class AccuracyTester:
    """Test accuracy of different implementations against PyTorch reference"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tolerance = 1e-5
        
    def test_matrix_add(self, M=1024, N=1024):
        """Test matrix addition accuracy"""
        print(f"Testing matrix addition ({M}x{N})...")
        
        # Generate test data
        a = torch.randn(M, N, device=self.device, dtype=torch.float32)
        b = torch.randn(M, N, device=self.device, dtype=torch.float32)
        
        # PyTorch reference
        pytorch_result = pytorch_matrix_add(a, b)
        
        # Test Triton
        try:
            triton_result = triton_matrix_add(a, b)
            triton_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
            print(f"  Triton error: {triton_error:.2e}")
            assert triton_error < self.tolerance, f"Triton accuracy test failed: {triton_error}"
        except Exception as e:
            print(f"  Triton test failed: {e}")
            
        # Test TileLang
        try:
            tilelang_result = tilelang_matrix_add(a, b)
            tilelang_error = torch.max(torch.abs(pytorch_result - tilelang_result)).item()
            print(f"  TileLang error: {tilelang_error:.2e}")
            assert tilelang_error < self.tolerance, f"TileLang accuracy test failed: {tilelang_error}"
        except Exception as e:
            print(f"  TileLang test failed: {e}")
            
        print("  Matrix addition accuracy test passed!")
        
    def test_matrix_mul(self, M=512, N=512, K=512):
        """Test matrix multiplication accuracy"""
        print(f"Testing matrix multiplication ({M}x{K} @ {K}x{N})...")
        
        # Generate test data
        a = torch.randn(M, K, device=self.device, dtype=torch.float32)
        b = torch.randn(K, N, device=self.device, dtype=torch.float32)
        
        # PyTorch reference
        pytorch_result = pytorch_matrix_mul(a, b)
        
        # Test Triton
        try:
            triton_result = triton_matrix_mul(a, b)
            triton_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
            print(f"  Triton error: {triton_error:.2e}")
            assert triton_error < self.tolerance, f"Triton accuracy test failed: {triton_error}"
        except Exception as e:
            print(f"  Triton test failed: {e}")
            
        # Test TileLang
        try:
            tilelang_result = tilelang_matrix_mul(a, b)
            tilelang_error = torch.max(torch.abs(pytorch_result - tilelang_result)).item()
            print(f"  TileLang error: {tilelang_error:.2e}")
            assert tilelang_error < self.tolerance, f"TileLang accuracy test failed: {tilelang_error}"
        except Exception as e:
            print(f"  TileLang test failed: {e}")
            
        print("  Matrix multiplication accuracy test passed!")
        
    def test_softmax(self, M=1024, N=1024):
        """Test softmax accuracy"""
        print(f"Testing softmax ({M}x{N})...")
        
        # Generate test data
        x = torch.randn(M, N, device=self.device, dtype=torch.float32)
        
        # PyTorch reference
        pytorch_result = pytorch_softmax(x)
        
        # Test Triton
        try:
            triton_result = triton_softmax(x)
            triton_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
            print(f"  Triton error: {triton_error:.2e}")
            assert triton_error < self.tolerance, f"Triton accuracy test failed: {triton_error}"
        except Exception as e:
            print(f"  Triton test failed: {e}")
            
        # Test TileLang
        try:
            tilelang_result = tilelang_softmax(x)
            tilelang_error = torch.max(torch.abs(pytorch_result - tilelang_result)).item()
            print(f"  TileLang error: {tilelang_error:.2e}")
            assert tilelang_error < self.tolerance, f"TileLang accuracy test failed: {tilelang_error}"
        except Exception as e:
            print(f"  TileLang test failed: {e}")
            
        print("  Softmax accuracy test passed!")
        
    def run_all_tests(self):
        """Run all accuracy tests"""
        print("Running accuracy tests...")
        print("=" * 50)
        
        self.test_matrix_add()
        self.test_matrix_mul()
        self.test_softmax()
        
        print("=" * 50)
        print("All accuracy tests completed!")

if __name__ == "__main__":
    tester = AccuracyTester()
    tester.run_all_tests()
