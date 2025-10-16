"""
Core unified interface layer
"""

from .kernel_registry import KernelRegistry, get_kernel, get_available_backends, get_available_operations
from .kernel_runner import KernelRunner, run_kernel
from .accuracy_checker import AccuracyChecker, compare_tensors, compare_against_reference
from .performance_profiler import PerformanceProfiler, measure_time, benchmark_kernels
from .logger import get_logger

__all__ = [
    'KernelRegistry',
    'get_kernel',
    'get_available_backends',
    'get_available_operations',
    'KernelRunner',
    'run_kernel',
    'AccuracyChecker',
    'compare_tensors',
    'compare_against_reference',
    'PerformanceProfiler',
    'measure_time',
    'benchmark_kernels',
    'get_logger'
]
