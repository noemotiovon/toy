import torch

class PyTorchMatrixMul:
    """PyTorch implementation for matrix multiplication"""
    
    @staticmethod
    def matrix_mul(a, b):
        """
        Matrix multiplication using PyTorch operations.
        This serves as the reference implementation for accuracy comparison.
        """
        return torch.matmul(a, b)
    
    @staticmethod
    def matrix_mul_bmm(a, b):
        """Batch matrix multiplication"""
        return torch.bmm(a, b)

class PyTorchOps:
    """Wrapper class for PyTorch operations"""
    
    @staticmethod
    def matrix_mul(a, b):
        return PyTorchMatrixMul.matrix_mul(a, b)
