"""
Kernel runner for executing kernels with error handling and validation
"""

import torch
from typing import Any, Dict, Optional, Tuple
from .kernel_registry import get_kernel
from .logger import get_logger

logger = get_logger(__name__)

class KernelRunner:
    """Runner for executing kernels with proper error handling"""
    
    def __init__(self, backend: str, operation: str):
        self.backend = backend
        self.operation = operation
        self.kernel = get_kernel(backend, operation)
        logger.debug(f"Initialized kernel runner for {backend}.{operation}")
    
    def run(self, *args, **kwargs) -> torch.Tensor:
        """Run the kernel with error handling"""
        try:
            # Validate inputs
            self._validate_inputs(*args)
            
            # Run kernel
            result = self.kernel(*args, **kwargs)
            
            # Validate output
            self._validate_output(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running {self.backend}.{self.operation}: {e}")
            raise
    
    def _validate_inputs(self, *args):
        """Validate input tensors"""
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                if not arg.is_contiguous():
                    logger.warning(f"Input tensor {i} is not contiguous, will be made contiguous")
                    args[i].data = arg.contiguous()
                
                if arg.device.type == 'cuda' and not torch.cuda.is_available():
                    raise RuntimeError("CUDA tensor provided but CUDA not available")
    
    def _validate_output(self, output: torch.Tensor):
        """Validate output tensor"""
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor output, got {type(output)}")
        
        if output.isnan().any():
            raise ValueError("Output contains NaN values")
        
        if output.isinf().any():
            raise ValueError("Output contains infinite values")

def run_kernel(backend: str, operation: str, *args, **kwargs) -> torch.Tensor:
    """Convenience function to run a kernel"""
    runner = KernelRunner(backend, operation)
    return runner.run(*args, **kwargs)
