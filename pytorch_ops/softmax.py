import torch

class PyTorchSoftmax:
    """PyTorch implementation for softmax"""
    
    @staticmethod
    def softmax(x, dim=-1):
        """
        Softmax using PyTorch operations.
        This serves as the reference implementation for accuracy comparison.
        """
        return torch.softmax(x, dim=dim)
    
    @staticmethod
    def softmax_stable(x, dim=-1):
        """Numerically stable softmax implementation"""
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x_shifted = x - x_max
        x_exp = torch.exp(x_shifted)
        x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        return x_exp / x_sum

class PyTorchOps:
    """Wrapper class for PyTorch operations"""
    
    @staticmethod
    def softmax(x, dim=-1):
        return PyTorchSoftmax.softmax(x, dim=dim)
