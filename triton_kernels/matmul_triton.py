import torch

def matmul_forward(a, b):
    """Matrix multiplication using PyTorch (as Triton placeholder)"""
    # For now, use PyTorch implementation as Triton has issues with matrix multiplication
    # This is a placeholder until we can fix the Triton implementation
    return torch.matmul(a, b)
