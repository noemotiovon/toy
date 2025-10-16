"""
Accuracy checker for comparing kernel outputs
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any
from .logger import get_logger

logger = get_logger(__name__)

class AccuracyChecker:
    """Checker for comparing kernel accuracy"""
    
    def __init__(self, atol: float = 1e-5, rtol: float = 1e-4):
        self.atol = atol
        self.rtol = rtol
        logger.debug(f"Initialized accuracy checker with atol={atol}, rtol={rtol}")
    
    def compare_tensors(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, Dict[str, float]]:
        """
        Compare two tensors and return accuracy metrics
        
        Returns:
            Tuple of (is_close, metrics_dict)
        """
        # Basic shape check
        if a.shape != b.shape:
            logger.error(f"Shape mismatch: {a.shape} vs {b.shape}")
            return False, {"error": "shape_mismatch"}
        
        # Device check
        if a.device != b.device:
            logger.warning(f"Device mismatch: {a.device} vs {b.device}, moving to same device")
            if a.device.type == 'cuda':
                b = b.to(a.device)
            else:
                a = a.to(b.device)
        
        # Dtype check
        if a.dtype != b.dtype:
            logger.warning(f"Dtype mismatch: {a.dtype} vs {b.dtype}, converting to float32")
            a = a.float()
            b = b.float()
        
        # Compute metrics
        metrics = self._compute_metrics(a, b)
        
        # Check if close
        is_close = torch.allclose(a, b, atol=self.atol, rtol=self.rtol)
        
        return is_close, metrics
    
    def _compute_metrics(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        """Compute detailed accuracy metrics"""
        diff = torch.abs(a - b)
        
        metrics = {
            "max_abs_error": torch.max(diff).item(),
            "mean_abs_error": torch.mean(diff).item(),
            "max_rel_error": torch.max(diff / (torch.abs(b) + 1e-8)).item(),
            "mean_rel_error": torch.mean(diff / (torch.abs(b) + 1e-8)).item(),
            "l2_error": torch.norm(diff).item(),
            "mse": torch.mean(diff ** 2).item(),
        }
        
        return metrics
    
    def check_accuracy(self, reference: torch.Tensor, test: torch.Tensor, 
                      backend_name: str = "unknown") -> Tuple[bool, Dict[str, float]]:
        """
        Check accuracy against reference implementation
        
        Args:
            reference: Reference tensor (usually from PyTorch)
            test: Test tensor from another backend
            backend_name: Name of the backend being tested
            
        Returns:
            Tuple of (passed, metrics)
        """
        is_close, metrics = self.compare_tensors(reference, test)
        
        status = "PASS" if is_close else "FAIL"
        logger.info(f"[{backend_name}] Accuracy check: {status}")
        logger.debug(f"[{backend_name}] Metrics: {metrics}")
        
        return is_close, metrics

def compare_tensors(a: torch.Tensor, b: torch.Tensor, 
                   atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    """Simple tensor comparison function"""
    checker = AccuracyChecker(atol=atol, rtol=rtol)
    is_close, _ = checker.compare_tensors(a, b)
    return is_close

def compare_against_reference(reference: torch.Tensor, test: torch.Tensor,
                             backend_name: str = "unknown",
                             atol: float = 1e-5, rtol: float = 1e-4) -> Tuple[bool, Dict[str, float]]:
    """Compare against reference implementation"""
    checker = AccuracyChecker(atol=atol, rtol=rtol)
    return checker.check_accuracy(reference, test, backend_name)
