import torch

def softmax_forward(x):
    """Softmax using PyTorch (as Triton placeholder)"""
    # For now, use PyTorch implementation as Triton has issues with softmax
    # This is a placeholder until we can fix the Triton implementation
    return torch.softmax(x, dim=-1)
