#!/usr/bin/env python3
"""
Comprehensive benchmark script for all kernels
"""

import torch
import argparse
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core import get_kernel, get_available_backends, benchmark_kernels, compare_against_reference
from core.logger import get_logger

logger = get_logger(__name__)

def create_test_data(operation: str, device: str = "cuda"):
    """Create test data for different operations"""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    if operation == "matrix_add":
        M, N = 2048, 2048
        a = torch.randn(M, N, device=device, dtype=torch.float32)
        b = torch.randn(M, N, device=device, dtype=torch.float32)
        return a, b
    elif operation == "matmul":
        M, K, N = 1024, 1024, 1024
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        return a, b
    elif operation == "softmax":
        M, N = 2048, 2048
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        return x
    else:
        raise ValueError(f"Unknown operation: {operation}")

def run_accuracy_tests():
    """Run accuracy tests for all kernels"""
    logger.info("Running accuracy tests...")
    
    operations = ["matrix_add", "matmul", "softmax"]
    results = {}
    
    for operation in operations:
        logger.info(f"Testing {operation} accuracy...")
        
        # Create test data
        test_data = create_test_data(operation)
        
        # Get reference (PyTorch)
        try:
            reference_func = get_kernel("torch", operation)
            reference = reference_func(*test_data) if isinstance(test_data, tuple) else reference_func(test_data)
        except Exception as e:
            logger.error(f"Failed to get reference for {operation}: {e}")
            continue
        
        # Test other backends
        backends = list(get_available_backends())
        backends.remove("torch")  # torch is our reference
        
        operation_results = {}
        for backend in backends:
            try:
                func = get_kernel(backend, operation)
                result = func(*test_data) if isinstance(test_data, tuple) else func(test_data)
                
                passed, metrics = compare_against_reference(reference, result, f"{backend}.{operation}")
                operation_results[backend] = {
                    "passed": passed,
                    "metrics": metrics
                }
                
            except Exception as e:
                logger.error(f"Failed to test {backend}.{operation}: {e}")
                operation_results[backend] = {"passed": False, "error": str(e)}
        
        results[operation] = operation_results
    
    return results

def run_performance_tests():
    """Run performance tests for all kernels"""
    logger.info("Running performance tests...")
    
    operations = ["matrix_add", "matmul", "softmax"]
    results = {}
    
    for operation in operations:
        logger.info(f"Benchmarking {operation}...")
        
        # Create test data
        test_data = create_test_data(operation)
        
        # Get all available kernels
        kernel_funcs = {}
        for backend in get_available_backends():
            try:
                kernel_funcs[backend] = get_kernel(backend, operation)
            except Exception as e:
                logger.warning(f"Failed to get {backend}.{operation}: {e}")
        
        if not kernel_funcs:
            logger.warning(f"No kernels available for {operation}")
            continue
        
        # Benchmark
        if isinstance(test_data, tuple):
            operation_results = benchmark_kernels(kernel_funcs, *test_data)
        else:
            operation_results = benchmark_kernels(kernel_funcs, test_data)
        
        results[operation] = operation_results
    
    return results

def print_results(accuracy_results, performance_results):
    """Print benchmark results in a nice format"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Accuracy results
    print("\nACCURACY TESTS:")
    print("-" * 40)
    for operation, results in accuracy_results.items():
        print(f"\n{operation.upper()}:")
        for backend, result in results.items():
            if "error" in result:
                print(f"  {backend:12}: ERROR - {result['error']}")
            else:
                status = "PASS" if result["passed"] else "FAIL"
                max_error = result["metrics"]["max_abs_error"]
                print(f"  {backend:12}: {status:4} (max error: {max_error:.2e})")
    
    # Performance results
    print("\nPERFORMANCE TESTS:")
    print("-" * 40)
    for operation, results in performance_results.items():
        print(f"\n{operation.upper()}:")
        # Sort by performance (best first)
        sorted_results = sorted(results.items(), key=lambda x: x[1].get("mean_time_ms", float('inf')))
        
        for backend, result in sorted_results:
            if not result["success"]:
                print(f"  {backend:12}: ERROR - {result['error']}")
            else:
                time_ms = result["mean_time_ms"]
                print(f"  {backend:12}: {time_ms:8.3f} ms")

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive kernel benchmarks")
    parser.add_argument("--accuracy-only", action="store_true", help="Run only accuracy tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    logger.info("Starting comprehensive benchmark...")
    logger.info(f"Available backends: {get_available_backends()}")
    
    accuracy_results = {}
    performance_results = {}
    
    # Run tests based on arguments
    if not args.performance_only:
        accuracy_results = run_accuracy_tests()
    
    if not args.accuracy_only:
        performance_results = run_performance_tests()
    
    # Print results
    print_results(accuracy_results, performance_results)
    
    logger.info("Benchmark completed!")

if __name__ == "__main__":
    main()
