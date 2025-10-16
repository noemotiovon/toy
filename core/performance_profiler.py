"""
Performance profiler for benchmarking kernels
"""

import time
import torch
import statistics
from typing import Dict, List, Callable, Any, Tuple
from .logger import get_logger

logger = get_logger(__name__)

class PerformanceProfiler:
    """Profiler for measuring kernel performance"""
    
    def __init__(self, warmup_runs: int = 10, test_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        logger.debug(f"Initialized profiler with warmup={warmup_runs}, test={test_runs}")
    
    def measure_time(self, func: Callable, *args, **kwargs) -> Tuple[float, Dict[str, float]]:
        """
        Measure execution time of a function
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Tuple of (mean_time, metrics_dict)
        """
        # Warmup runs
        logger.debug(f"Running {self.warmup_runs} warmup iterations")
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error during warmup: {e}")
                raise
        
        # Synchronize GPU if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Actual timing
        times = []
        logger.debug(f"Running {self.test_runs} timing iterations")
        
        for i in range(self.test_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error during timing iteration {i}: {e}")
                raise
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Compute statistics
        metrics = self._compute_stats(times)
        
        return metrics["mean"], metrics
    
    def _compute_stats(self, times: List[float]) -> Dict[str, float]:
        """Compute timing statistics"""
        if not times:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        
        return {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "median": statistics.median(times),
            "total": sum(times),
        }
    
    def benchmark_kernel(self, kernel_func: Callable, *args, 
                        backend_name: str = "unknown", **kwargs) -> Dict[str, Any]:
        """
        Benchmark a kernel and return comprehensive results
        
        Args:
            kernel_func: Kernel function to benchmark
            *args: Arguments for the kernel
            backend_name: Name of the backend
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking {backend_name} kernel")
        
        try:
            mean_time, timing_metrics = self.measure_time(kernel_func, *args, **kwargs)
            
            # Get output for shape/dtype info
            output = kernel_func(*args, **kwargs)
            
            result = {
                "backend": backend_name,
                "mean_time_ms": mean_time * 1000,
                "timing_metrics": timing_metrics,
                "output_shape": output.shape if isinstance(output, torch.Tensor) else None,
                "output_dtype": output.dtype if isinstance(output, torch.Tensor) else None,
                "success": True,
                "error": None,
            }
            
            logger.info(f"[{backend_name}] Mean time: {mean_time*1000:.3f} ms")
            
        except Exception as e:
            logger.error(f"[{backend_name}] Benchmark failed: {e}")
            result = {
                "backend": backend_name,
                "mean_time_ms": float('inf'),
                "timing_metrics": {},
                "output_shape": None,
                "output_dtype": None,
                "success": False,
                "error": str(e),
            }
        
        return result

def measure_time(func: Callable, *args, warmup: int = 10, iterations: int = 100, **kwargs) -> float:
    """Simple timing function"""
    profiler = PerformanceProfiler(warmup_runs=warmup, test_runs=iterations)
    mean_time, _ = profiler.measure_time(func, *args, **kwargs)
    return mean_time

def benchmark_kernels(kernel_funcs: Dict[str, Callable], *args, **kwargs) -> Dict[str, Dict[str, Any]]:
    """Benchmark multiple kernels"""
    profiler = PerformanceProfiler()
    results = {}
    
    for backend_name, kernel_func in kernel_funcs.items():
        results[backend_name] = profiler.benchmark_kernel(kernel_func, *args, 
                                                          backend_name=backend_name, **kwargs)
    
    return results
