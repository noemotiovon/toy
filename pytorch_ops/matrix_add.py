import torch

class PyTorchMatrixAdd:
    """PyTorch implementation for matrix addition"""
    
    @staticmethod
    def matrix_add(a, b):
        """
        Matrix addition using PyTorch operations.
        This serves as the reference implementation for accuracy comparison.
        """
        return torch.add(a, b)
    
    @staticmethod
    def matrix_add_inplace(a, b):
        """In-place matrix addition"""
        return a.add_(b)

class PyTorchOps:
    """Wrapper class for PyTorch operations"""
    
    @staticmethod
    def matrix_add(a, b):
        return PyTorchMatrixAdd.matrix_add(a, b)
