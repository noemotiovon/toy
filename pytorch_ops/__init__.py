# PyTorch operator implementations
from .matrix_add import PyTorchOps as MatrixAddOps
from .matrix_mul import PyTorchOps as MatrixMulOps
from .softmax import PyTorchOps as SoftmaxOps

def matrix_add(a, b):
    return MatrixAddOps.matrix_add(a, b)

def matrix_mul(a, b):
    return MatrixMulOps.matrix_mul(a, b)

def softmax(x, dim=-1):
    return SoftmaxOps.softmax(x, dim=dim)

__all__ = ['matrix_add', 'matrix_mul', 'softmax']