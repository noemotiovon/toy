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

def softmax_forward(x):
    """Softmax using PyTorch"""
    return PyTorchSoftmax.softmax(x, dim=-1)
