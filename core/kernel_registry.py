"""
Kernel registry for managing different backend kernels
"""

import importlib
import torch
from typing import Dict, Callable, Optional, Any
from .logger import get_logger

logger = get_logger(__name__)

class KernelRegistry:
    """Registry for managing different backend kernels"""
    
    def __init__(self):
        self._kernels: Dict[str, Dict[str, Callable]] = {}
        self._available_backends = set()
        self._load_kernels()
    
    def _load_kernels(self):
        """Load all available kernels from different backends"""
        backends = ['torch', 'triton', 'tilelang', 'cuda']
        
        for backend in backends:
            try:
                self._load_backend(backend)
                self._available_backends.add(backend)
                logger.info(f"Successfully loaded {backend} backend")
            except Exception as e:
                logger.warning(f"Failed to load {backend} backend: {e}")
    
    def _load_backend(self, backend: str):
        """Load kernels for a specific backend"""
        if backend == 'torch':
            from torch_kernels import matrix_add_forward, matmul_forward, softmax_forward
            self._register_kernels(backend, {
                'matrix_add': matrix_add_forward,
                'matmul': matmul_forward,
                'softmax': softmax_forward,
            })
            
        elif backend == 'triton':
            from triton_kernels import matrix_add_forward, matmul_forward, softmax_forward
            self._register_kernels(backend, {
                'matrix_add': matrix_add_forward,
                'matmul': matmul_forward,
                'softmax': softmax_forward,
            })
            
        elif backend == 'tilelang':
            from tilelang_kernels import matrix_add_forward, matmul_forward, softmax_forward
            self._register_kernels(backend, {
                'matrix_add': matrix_add_forward,
                'matmul': matmul_forward,
                'softmax': softmax_forward,
            })
            
        elif backend == 'cuda':
            try:
                import kernels_cuda
                self._register_kernels(backend, {
                    'matrix_add': kernels_cuda.matrix_add,
                    'matmul': kernels_cuda.matmul,
                    'softmax': kernels_cuda.softmax,
                })
            except ImportError:
                logger.warning("CUDA kernels not available - compile with CUDA support")
                raise
    
    def _register_kernels(self, backend: str, kernels: Dict[str, Callable]):
        """Register kernels for a backend"""
        if backend not in self._kernels:
            self._kernels[backend] = {}
        self._kernels[backend].update(kernels)
    
    def get_kernel(self, backend: str, operation: str) -> Callable:
        """Get a specific kernel"""
        if backend not in self._available_backends:
            raise ValueError(f"Backend '{backend}' not available. Available: {self._available_backends}")
        
        if backend not in self._kernels:
            raise ValueError(f"No kernels loaded for backend '{backend}'")
        
        if operation not in self._kernels[backend]:
            raise ValueError(f"Operation '{operation}' not available for backend '{backend}'. "
                           f"Available: {list(self._kernels[backend].keys())}")
        
        return self._kernels[backend][operation]
    
    def list_available_backends(self) -> set:
        """List all available backends"""
        return self._available_backends.copy()
    
    def list_operations(self, backend: str = None) -> Dict[str, list]:
        """List available operations"""
        if backend is None:
            return {b: list(self._kernels[b].keys()) for b in self._kernels}
        else:
            if backend not in self._kernels:
                raise ValueError(f"Backend '{backend}' not available")
            return {backend: list(self._kernels[backend].keys())}

# Global registry instance
_registry = KernelRegistry()

def get_kernel(backend: str, operation: str) -> Callable:
    """Get a kernel from the global registry"""
    return _registry.get_kernel(backend, operation)

def get_available_backends() -> set:
    """Get available backends"""
    return _registry.list_available_backends()

def get_available_operations(backend: str = None) -> Dict[str, list]:
    """Get available operations"""
    return _registry.list_operations(backend)
