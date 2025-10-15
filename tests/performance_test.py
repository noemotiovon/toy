import torch
import time
import numpy as np
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_ops import matrix_add as pytorch_matrix_add, matrix_mul as pytorch_matrix_mul, softmax as pytorch_softmax
from triton_ops import matrix_add as triton_matrix_add, matrix_mul as triton_matrix_mul, softmax as triton_softmax
from tilelang_ops import matrix_add as tilelang_matrix_add, matrix_mul as tilelang_matrix_mul, softmax as tilelang_softmax

class PerformanceTester:
    """Test performance of different implementations"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', warmup_runs=10, test_runs=100):
        self.device = device
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.results = {}
        
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function with warmup and multiple runs"""
        # Warmup
        for _ in range(self.warmup_runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            func(*args, **kwargs)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        # Actual timing
        times = []
        for _ in range(self.test_runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            result = func(*args, **kwargs)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            
        return np.mean(times), np.std(times), result
    
    def test_matrix_add_performance(self, sizes=[(1024, 1024), (2048, 2048), (4096, 4096)]):
        """Test matrix addition performance"""
        print("Testing matrix addition performance...")
        results = {}
        
        for M, N in sizes:
            print(f"  Testing {M}x{N}...")
            a = torch.randn(M, N, device=self.device, dtype=torch.float32)
            b = torch.randn(M, N, device=self.device, dtype=torch.float32)
            
            # PyTorch
            try:
                mean_time, std_time, _ = self.benchmark_function(pytorch_matrix_add, a, b)
                results[f'PyTorch_{M}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    PyTorch: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    PyTorch failed: {e}")
                
            # Triton
            try:
                mean_time, std_time, _ = self.benchmark_function(triton_matrix_add, a, b)
                results[f'Triton_{M}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    Triton: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    Triton failed: {e}")
                
            # TileLang
            try:
                mean_time, std_time, _ = self.benchmark_function(tilelang_matrix_add, a, b)
                results[f'TileLang_{M}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    TileLang: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    TileLang failed: {e}")
                
        self.results['matrix_add'] = results
        
    def test_matrix_mul_performance(self, sizes=[(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]):
        """Test matrix multiplication performance"""
        print("Testing matrix multiplication performance...")
        results = {}
        
        for M, K, N in sizes:
            print(f"  Testing {M}x{K} @ {K}x{N}...")
            a = torch.randn(M, K, device=self.device, dtype=torch.float32)
            b = torch.randn(K, N, device=self.device, dtype=torch.float32)
            
            # PyTorch
            try:
                mean_time, std_time, _ = self.benchmark_function(pytorch_matrix_mul, a, b)
                results[f'PyTorch_{M}x{K}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    PyTorch: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    PyTorch failed: {e}")
                
            # Triton
            try:
                mean_time, std_time, _ = self.benchmark_function(triton_matrix_mul, a, b)
                results[f'Triton_{M}x{K}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    Triton: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    Triton failed: {e}")
                
            # TileLang
            try:
                mean_time, std_time, _ = self.benchmark_function(tilelang_matrix_mul, a, b)
                results[f'TileLang_{M}x{K}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    TileLang: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    TileLang failed: {e}")
                
        self.results['matrix_mul'] = results
        
    def test_softmax_performance(self, sizes=[(1024, 1024), (2048, 2048), (4096, 4096)]):
        """Test softmax performance"""
        print("Testing softmax performance...")
        results = {}
        
        for M, N in sizes:
            print(f"  Testing {M}x{N}...")
            x = torch.randn(M, N, device=self.device, dtype=torch.float32)
            
            # PyTorch
            try:
                mean_time, std_time, _ = self.benchmark_function(pytorch_softmax, x)
                results[f'PyTorch_{M}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    PyTorch: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    PyTorch failed: {e}")
                
            # Triton
            try:
                mean_time, std_time, _ = self.benchmark_function(triton_softmax, x)
                results[f'Triton_{M}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    Triton: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    Triton failed: {e}")
                
            # TileLang
            try:
                mean_time, std_time, _ = self.benchmark_function(tilelang_softmax, x)
                results[f'TileLang_{M}x{N}'] = {'mean': mean_time, 'std': std_time}
                print(f"    TileLang: {mean_time*1000:.2f}±{std_time*1000:.2f} ms")
            except Exception as e:
                print(f"    TileLang failed: {e}")
                
        self.results['softmax'] = results
        
    def run_all_tests(self):
        """Run all performance tests"""
        print("Running performance tests...")
        print("=" * 50)
        
        self.test_matrix_add_performance()
        self.test_matrix_mul_performance()
        self.test_softmax_performance()
        
        print("=" * 50)
        print("All performance tests completed!")
        
        return self.results

if __name__ == "__main__":
    tester = PerformanceTester()
    results = tester.run_all_tests()
